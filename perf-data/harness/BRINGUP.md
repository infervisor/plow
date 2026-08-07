# LLM bring-up harness

The single runbook for bringing a new model (network) up on plow — on a new GPU
or a known one — and driving it to a defensible performance number. Merged from
two campaigns:

- **Gemma-4-12B / RTX 5090 (sm_120a)** — decode/serving bring-up; measurements
  in `perf-data/gemma4-12b-sm120-serving.md`.
- **Gemma-4-12B / GH200 (sm_90a)** — prefill parity campaign (fp8 6.8x behind
  vLLM → statistical parity); measurements in
  `perf-data/gemma12b-gh200-prefill-campaign.md`.

The ordering is the point. **The interpreter must be optimal and every kernel
arm the geometry needs must be present in the cubin BEFORE any blob is emitted
or any number is quoted.** A missing arm is not a crash — it is a silently
wrong answer or a silent fallback, and every stage below is designed to fail
loudly instead.

---

## Ground rules (non-negotiable, every stage)

1. **Every GPU run goes through `perf-data/harness/gpulease <label> <cmd>`.**
   Exit 76 = the GPU was contended; the run's timings are untrustworthy —
   re-run. gpulease audits for foreign compute processes before/after and warns
   loudly. Never quote a number from a run that printed the contention warning.
2. **Correctness gates before performance numbers, always.** No perf cell is
   recorded for an arm that has not passed the token-identity gate
   (`bringup_gate.sh`) against the reference arm that same build. A kernel that
   is fast and wrong is a wrong kernel.
3. **A/B under the same conditions or not at all.** Arms benched under
   different contention are not comparable. When background load changes
   (a foreign server appears or exits), re-baseline EVERYTHING.
4. **THE PROBE LAW: standalone kernel probes OVERSTATE.** A probe re-runs one
   kernel on hot inputs — its operands and outputs become L2-resident in ways a
   real forward pass never sees. The GH200 campaign had a +15% probe win that
   was -3 ms in-model, and a "2.7x win" shape fallback that was -8 ms in-model
   (co-scheduled ops change the picture). The sm_120a campaign had the same
   lesson from the other side: an isolated GEMV microbench scored
   `GV_MM_MAX=16` as a 1.00x no-op where the megakernel measured 1.43x. A probe
   may motivate a variant or *prune* a bad one; only the in-model gate+bench
   decides whether it lands. No exceptions.

---

## Stage 0 — environment sanity (do this first, it invalidates everything else)

| check | command | why |
|-------|---------|-----|
| driver vs libcuda | `nvidia-smi --query-gpu=driver_version --format=csv` | a CUDA **compat** libcuda NEWER than the driver fails `cuInit` |
| GPU actually used | grep the serve log for `backend ready — GPU accelerated` | CPU fallback is a **WARNING**, not an error |
| toolchain | `nix develop` | cargo/rustc are not on the default PATH |
| contention | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | resident foreign servers poison every timing (rule 1/3) |

    # If cuInit fails with CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
    export PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1

**Trap.** Without `PLOW_LIBCUDA` pointing at the system lib, plowrt drops to the
CPU reference interpreter and still serves correct text — slowly. An unwary
benchmark measures the CPU path. Always confirm the `GPU accelerated` line.

Profiler availability differs by tool: `sudo ncu` WORKS (perf-counter
permission via sudo); `nsys` does NOT intercept plowrt (it dlopens libcuda).
Check both once per box before you need them.

---

## Stage 1 — geometry audit (decides which kernels you need)

Read `config.json` and write these down. They select kernel arms and set the
VRAM budget; everything downstream depends on them.

| field | why it matters |
|-------|----------------|
| `layer_types` (full vs sliding count) | selects flash arms; sliding layers get a ring |
| `head_dim` per family | heterogeneous hd (e.g. 256 sliding / 512 full) forces per-HD template instantiations; hd128 models fall back to the mma.sync arms on sm_90a |
| `num_key_value_heads` per family | kvh=1 full layers are a different GEMV shape from kvh=8 |
| `sliding_window` | sets the KV ring floor — see the budget formula |
| `vocab_size` | lm_head GEMV + sampler scratch |
| `rope_parameters` sub-objects | per-family RoPE recipes (v7 blob) |
| MoE vs dense | entirely different op family (`op_moe.cuh`) |
| `num_hidden_layers` | segment count scales ~10x layers; the loader cap is 2048 (raised from 512 when 60-layer Gemma-31B tripped it) |

Also check the GEMM tile divisibility for the sm_90a n256 bodies: every
projection N should divide 256 and every K divide 128 (all shipped models do —
12B: 4096/512/3840/15360; 31B: 8192/4096/5376/21504; Qwen3/Llama dims likewise).
Odd N traps loudly; unmapped or non-conforming GEMMs fall to the fat object's
cp.async path via the classing (correct, slower).

Gemma-4-12B for reference: 48 layers = 8 full (kvh 1, hd 512) + 40 sliding
(kvh 8, hd 256), window 1024, vocab 262144, dense, bf16.

---

## Stage 2 — interpreter + kernel presence (the gate)

Build the cubins, then **prove** the arms exist. Do not skip the verification —
this is the stage that catches a geometry the objects were never compiled for.

sm_120a (single-object decode/prefill):

    PLOW_ROOT=$PWD scripts/build_sm120_cubin.sh <assets>/interp_sm120.cubin

sm_90a (the five-object segmented prefill stack — see the canonical
configuration section below for the full flag set):

    PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32" \
    PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_W8A8=1 PLOW_BUILD_SEG=1 \
    PLOW_BUILD_FATLITE=1 PLOW_BUILD_GEMM_WS384=1 \
    PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1 \
    scripts/build_sm90a_cubin.sh <assets>/interp_sm90a.cubin

Then verify:

    # 1. entry symbols present (the scripts already fail hard if not)
    cuobjdump -symbols <assets>/interp_sm120.cubin    | grep _Z12interp_sm12011PlowProgram
    cuobjdump -symbols <assets>/interp_sm120_pf.cubin | grep _Z15interp_sm120_pf11PlowProgram
    cuobjdump -symbols <assets>/sample_sm120.cubin    | grep -E 'plow_sample|plow_advance'

    # 2. register pressure / spill of every object
    cuobjdump -res-usage <assets>/interp_sm90a_pfgemm.cubin | grep -A1 pfgemm

`plow_advance` in the sampler object is what gates `PLOW_MULTISTEP`; if it is
missing, multi-step silently disables itself with a warning and you lose ~1.7x.

**Register reading.** An interpreter object's register count is the MAX over
every instantiated arm — a wide rung added for batching taxes the B=1 path too.
Counterintuitively, *more* spill sometimes measures faster (the sm_120a
`GV_UN16` sweep): latency-bound rungs tolerate spill. Do not assume
spill == slow; measure.

**The TU-isolation law (sm_90a, three times proven).** Heavyweight wgmma bodies
lose probe-grade register allocation when compiled into a wide-armed
interpreter TU: the warp-spec GEMM deadlocked/spilled in the fat TU and ran
clean in an arm-stripped one; the hd512 wgmma flash was 1.75-2.5x WORSE in the
fat TU and won 10.5% in a flash-only TU; the 384-thread producer/consumer GEMM
is only expressible as its own object. When a big kernel underperforms
in-model, suspect the TU before the kernel.

### Feature -D matrix (cubin layer)

Pass via `PLOW_EXTRA_DEFINES` (appended to every object by the build script):

| define | default | when to set |
|--------|---------|-------------|
| `PLOW_NV_GEMMA=1` | on in script | Gemma family arms |
| `PLOW_NV_FA_GF=2`, `PLOW_NV_FA_GF_FULL=4` | on in script | flash granularity, the committed recipe |
| `PLOW_NV_EMBED_SMEM=1` | on in script | embedding staging |
| `PLOW_FP8_KV=1` | OFF | e4m3 KV arms (`HEADNORM_ROPE_FP8`, `FLASH_DECODE_FP8`). Needed to fit large batch |
| `PLOW_NV_W8A8=1` | OFF | fp8 tensor-core PREFILL; required with fp8 weights |
| `PLOW_NV_FA_PIPE=0` | — | required for fp8 prefill (cp.async cannot convert fp8 inline) |
| `GV_MM_MAX=16` | 8 | **set when the blob's decode batch >= 16** — see Stage 5b |
| `GV_UN16=8` | 4 | wide-rung unroll; best measured on sm_120a |
| `PLOW_NV_FA256_BKV=64`, `PLOW_NV_FA512_BKV=32` | 32 / 16 | sm_90a flash KV-tile sizes (the measured optima) |

### Full cubin knob inventory

Every compile-time knob the NVIDIA objects read, with its in-source default.
Only the first group is normally touched; the rest are ablation/experiment
switches that exist so a finding can be re-tested, and should be left alone
unless you are reproducing one.

**Feature selection — you must set these to match the model and blob:**

| knob | default | meaning |
|------|---------|---------|
| `PLOW_NV_GEMMA` | 0 | Gemma-family op arms |
| `PLOW_NV_PREFILL` | 0 | builds the prefill object instead of decode |
| `PLOW_NV_MLA` | 1 | MLA (DeepSeek-style) attention arms |
| `PLOW_NV_MAMBA` | 1 | Mamba/SSM arms |
| `PLOW_NV_DSA` | 1 | DSA sparse-attention arms |
| `PLOW_FP8_KV` | 0 | e4m3 KV cache arms |
| `PLOW_NV_W8A8` | 0 | fp8 tensor-core prefill mainloop |
| `PLOW_NV_EMBED_SMEM` | 0 | embedding staged through smem |
| `PLOW_NV_GF8_TWIN` | 0 | GQA-fused hd512 twin |

**sm_90a segmented-object selection (the shipping prefill stack; set via the
build script's `PLOW_BUILD_*` envs rather than raw -D):**

| build env | object it shapes | meaning |
|-----------|------------------|---------|
| `PLOW_BUILD_SEG=1` | `_pfseg` + `_pfgemm` | the segmented pair |
| `PLOW_BUILD_FATLITE=1` | `_pfseg` | fat object arm-stripped of flash, 128-reg, occ-2 |
| `PLOW_BUILD_GEMM_WS384=1` | `_pfgemm` | 384-thread producer/consumer GEMM (the cuBLAS shape; carries BOTH precisions' n256 bodies) |
| `PLOW_BUILD_FA512=1` + `FA_WG` + `FA_HD256` | `_pffa` | dedicated flash object, wgmma arms, hd256+hd512 |
| `PLOW_BUILD_TMA_GEMM=1` / `PLOW_BUILD_W8A8=1` | all | TMA GEMM bodies / fp8 w8a8 arms |

**Performance tuning — measured, safe to sweep:**

| knob | default | notes |
|------|---------|-------|
| `GV_MM_MAX` | 8 | decode GEMV rung ceiling; 16 at B>=16 (see Stage 5b) |
| `GV_UNROLL` / `GV_UNROLL_GLU` | 8 / 4 | base rung unroll |
| `GV_UN16` / `GV_UN_GLU16` | 4 / 2 | MM=16 rung unroll (8 measured best on sm_120a) |
| `GV_UN32` / `GV_UN_GLU32` | 2 / 1 | MM=32 rung unroll |
| `GV_UNROLL_FP8` / `GV_UNROLL_GLU_FP8` | =bf16 twins | fp8 rung unroll |
| `GV_MOE_RB` / `GV_MOE_UN` | 2 / 2 | MoE GEMV row-block / unroll |
| `PLOW_NV_FA_GF` | 4 | flash GQA-fusion granularity (script sets 2) |
| `PLOW_NV_FA_GF_FULL` | — | same for full-attention layers (script sets 4) |
| `PLOW_NV_FA_PIPE` | 1 | cp.async staging; **must be 0 for fp8 prefill** |
| `PLOW_NV_FA_KUN` / `PLOW_NV_FA_PX4` | 1 / 1 | flash K-unroll / PX4 arm |
| `PLOW_NV_GATE_SLEEP` | 64 | interpreter gate spin before sleep |
| `PLOW_NV_SCHED` / `PLOW_NV_PTXSYNC` | 1 / 1 | scheduler + PTX barrier arms |
| `PGM90_TILE_BAND` | 16 | sm_90a GEMM band rasterization (L2 B-tile sharing) |
| `PGM90_UNI256_NS` | 4 | sm_90a n256 TMA ring depth |

**Ablation / experiment switches — default OFF, leave off:**
`PLOW_NV_GEMV_RB`, `PLOW_NV_RB_GEMV`, `PLOW_NV_RB_QKV`, `PLOW_NV_RB_LMHEAD`,
`GV_RB`, `GV_UNROLL_RB`, `GV_LS_SG`, `PLOW_NV_GEMV_LS`,
`PLOW_NV_GEMV_NOSTAGE`, `PLOW_NV_GEMV_STAGE_MINROWS`, `PLOW_NV_FP8_RB`,
`PLOW_NV_FA_TMA`, `PLOW_NV_FA_WPR`, `PLOW_NV_FA_WPR_RB`, `PLOW_NV_FA_VDBUF`,
`PLOW_NV_FA_QGLOB`, `PLOW_NV_FA_CORRSKIP`, `PLOW_NV_FA_REDBOUND`,
`PLOW_NV_FA_FP8ABL`, `PLOW_NV_KVBOUNDS`, `PLOW_NV_LEAN_DECODE`,
`PLOW_NV_SKELETON`, `PLOW_NV_SKEL_PAD`, `PLOW_NV_TRACE`,
`PLOW_NV_ABLATE_LO/HI`, `PLOW_NV_SEG_WS`/`WS_ENTRY`, `PGM90_WS_BN256`,
`PLOW_NV_FA_WGITEM`, `PLOW_NV_SEG_OCC1`/`SEG_NOGLU`.

Note `PLOW_NV_MLA`, `PLOW_NV_MAMBA` and `PLOW_NV_DSA` default **ON**: the stock
object already carries those families, so a non-Gemma network does not need a
different build to have its arms present — only `PLOW_NV_GEMMA` and the
precision switches are model-specific.

---

## Stage 3 — blob emit (plowc) and the VRAM budget

Compute the budget BEFORE emitting; the planner refuses at startup otherwise.

    sliding_per_seq = ring x n_slide x kvh_slide x hd_slide x 2(k,v) x elt
    full_per_seq    = ctx  x n_full  x kvh_full  x hd_full  x 2(k,v) x elt
    ring            = min(ctx, next_pow2(window + PLOW_MAX_CHUNK - 1))
    total           = weights + batch x (sliding_per_seq + full_per_seq) + activations

**The ring is set by the prefill chunk, not the model.** This is why
`PLOW_MAX_CHUNK` now defaults to `next_pow2(window)` on a windowed model and
stays at 8192 when `window == 0`: at 8192 a window-1024 model rings 16384 rows,
an 8x inflation over what the window needs, purely because of the chunk.
Measured on Gemma-4-12B at B=16/ctx 8192, the old default wanted 21.32 GiB of
KV against 6.09 GiB now — it did not fit beside 12 GiB of weights on a 32 GiB
card. There is a hard floor at `next_pow2(window)`, so window 1024 cannot go
below 2048 rows however small the chunk.

Emit knobs (decode-oriented bring-up, sm_120a style):

    # NS_FULL_ABS and MAX_CHUNK now default correctly — only these are needed:
    PLOW_UNISEG=1 PLOW_DECODE_BATCH=<B> \
    [PLOW_FP8_KV=1] [PLOW_FP8=1 PLOW_W8A8=1] \
      plowc --hf-dir <model> --gpu <gpu> --emit devblob --max-ctx <ctx> \
            [--weight-dtype fp8] --out <dir>          # <dir>, NOT <dir>/model.pkt

For the sm_90a segmented prefill stack, use the canonical emit line in the
"Canonical sm_90a prefill configuration" section instead (NO `PLOW_UNISEG` —
the wave-class segments are the point; small buckets auto-collapse via
`PLOW_UNISEG_MAX_T`).

`--out` pointing at a `.pkt` path emits a bare blob (`bundle=false`); point it
at a DIRECTORY to get a servable bundle (checkpoint symlink, tokenizer,
weights.json).

**`PLOW_DECODE_BATCH` is a fixed kernel width, not a maximum.** A B=32 blob pays
the full B=32 step even when 16 slots are live — measured 165 tok/s at c16 vs
240 for a B=16 blob. Match B to expected concurrency.

**Watch the chunk ladder against your benchmark lengths.** The chat template
pushes a "4096-token" prompt to ~4110 rows; if the ladder splits it
`[4096, 128]`, the 128-token tail is a SECOND full-model pass (~30-36 ms
measured — weight restream + per-packet floors) that the competitor does not
pay. `PLOW_PF_LADDER_APPEND=640,1152,2176,4224` adds +128 overhang rungs at
each standard point; the runtime chunk-cost model picks them automatically.

### fp8 weights need a pre-quantized checkpoint

plow does not quantize on load; the runtime looks for `fp8/<name>` twins and
fails with `MISSING WEIGHT: fp8/...` if absent.

    python3 perf-data/harness/quantize_fp8.py <src-model-dir> <out-dir>
    # then a checkpoint dir holding BOTH (the loader globs *.safetensors):
    #   model.safetensors      -> original bf16 (norms/embeddings/lm_head)
    #   model-fp8.safetensors  -> the fp8 twins

---

## Stage 4 — correctness gate (before ANY performance number)

`bringup_gate.sh <assets-dir> <tag> <port> [plowrt-binary]` — serves the
bundle, runs 4 fixed prompts greedy (temperature 0, 32 tokens), dumps outputs
to `$BRINGUP_OUT/gate-out/<tag>.txt`. Compare with `diff` against the
reference arm's file. Classification:

- **Token-identical** — required for pure refactors, staging changes, work
  reordering, epilogue vectorization (quantize from the *rounded* value to keep
  identity through fusions).
- **Coherent-but-shifted** — acceptable only for documented numerics changes
  (wgmma reassociation, unpromoted fp8 accumulation, warp-vs-block reduction
  order), and must be called out in the commit.
- Garbage / truncation / wrong first token — reject. (The first generated token
  comes from the prefill argmax: a corrupt first token means the lm_head path
  is broken even when decode "recovers".)

Beyond the 4-prompt gate, run each of these once per bring-up — each catches a
distinct silent-wrong:

1. **Multi-chunk prefill** — a prompt longer than `PLOW_MAX_CHUNK` to exercise
   ring wraparound on sliding layers.
2. **Stop behavior** — confirm generation stops at eos, and that
   `"ignore_eos": true` runs to `max_completion_tokens` with
   `finish_reason: "length"`.
3. **Cap honored** — send `max_completion_tokens` and confirm the response
   respects it (OpenAI renamed `max_tokens`; a server binding only the old name
   silently runs to eos).
4. **Lossy modes are opt-in** — fp8 KV diverges from greedy bf16 after ~21
   tokens by design. Never compare a lossy plow config against a bf16 baseline
   without saying so.

---

## Stage 5 — performance bring-up ladder

### 5.0 Attribution first (cheapest first; drill down only when a level is flat)

1. **Client vs server**: `PLOW_TTFT_LOG=1` on the server dumps the TTFT
   breakdown per request (template / tokenize / queue / prefill / detok /
   unaccounted-HTTP). Both CUDA and AMD arms are instrumented. This is how the
   30 ms tail-chunk and the 11 ms tokenizer were found on GH200.
2. **Per-class wall** (sm_90a segmented prefill): `PLOW_PF_SEG_TIME=1` wraps
   every segment launch in CUDA events and logs per-class totals (GEMM / fat /
   flash) plus the 10 slowest segments. Event overhead perturbs (~5%); use for
   shares, not absolutes.
3. **Per-op**: build cubins with `PLOW_EXTRA_DEFINES=-DPLOW_NV_TRACE=1`, serve
   with `PLOW_PF_TRACE_LOG=1` — block-0 gate/body/signal cycles by opcode.
   Block 0 undercounts imbalanced ops ~3x; again shares, not absolutes.
4. **Decode step split**: `PLOW_STEP_TIME=1` prints
   `gap_us=<G> dev_interp_ms=<K>` — host gap vs device kernel. Branch:
   `G >> K` → host-bound (5a); `K >> G` → kernel-bound (5b); comparable → do
   5a first (it is free).
5. **Kernel-level**: `sudo ncu --set full` + Warp State Statistics for the
   binding stall (`nsys` cannot intercept the dlopen'd driver).
6. **Clock sanity**: sample `nvidia-smi --query-gpu=clocks.sm,power.draw,
   clocks_throttle_reasons.active -lms 500` during a bench before blaming
   silicon — GH200 held 1980 MHz flat at 470 W.

### 5.0b Roofline sanity before writing any kernel variant

Do the arithmetic FIRST, then measure the box ceiling:

- Candidate limiters, in the order they actually bound the GH200 campaign:
  per-launch floors (~60-90 µs x segment count), smem bandwidth (~128 B/cyc/SM:
  TMA writes + wgmma SS reads both count), TMA per-SM write rate, tensor-core
  rate at the tile shape, DRAM (weights are read once per chunk — usually NOT
  the limiter), L2 service (refuted as a limiter — multicast made it slower).
- **Measure the practical ceiling with cuBLASLt/torch at the EXACT shapes**
  (`bringup_ceiling.py`). On GH200: fp8 1324-1468 TF/s, bf16 804-861 at the
  12B shapes — knowing this is what justified the 384-thread object and ended
  three dead-end micro-knob rounds.

### 5a. Host-bound (decode)

Device sampling and multi-step are now **ON by default** (`PLOW_DEV_SAMPLE=0` /
`PLOW_MULTISTEP=0` opt out); confirm both lines appear in the serve log. Add:

    PLOW_PF_BATCH=1       # cross-request batched prefill (still opt-in)

Measured on Gemma-4-12B at B=16: gap 83.6 -> 32.2 ms, throughput 102.74 ->
185.60 tok/s. **This was the single largest free win, which is why it is now
the default.** K=8 captures nearly all of it; K=32 adds ~3%.

`PLOW_PF_INTERLEAVE` / `PLOW_PF_CHUNK` measured at or below their defaults on
this workload — leave them alone unless the gap is provably prefill.

### 5b. Kernel-bound (decode)

Ladder, biggest first:

1. **`GV_MM_MAX`** — the decode GEMV ladder is `{1,2,4,8}` and walks M in blocks
   of 8 above that, so **B=16 streams the entire weight set twice**. Measured
   B=16 = 1.85x B=8 for identical weights. `-DGV_MM_MAX=16` took the B=16 kernel
   41.17 -> 28.8 ms. Note it *hurts* B=8 (355 -> 294 tok/s) — set only for B>=16.
2. **`PLOW_NS_FULL_ABS`** — decode split factor. Monotone at B=16 on 170 SMs:
   8 -> 25.18 ms, 16 -> 25.28, 32 -> 25.93, 48 -> 26.32.
3. **`GV_UN16` / `GV_UN_GLU16`** — wide-rung unroll, ~2%. Precision-dependent:
   sweep at the precision you ship.
4. **Weight dtype** — fp8 weights cut 22.2 -> 12.0 GiB but only 28.8 -> 26.5 ms
   (dequant-compute-bound). Take fp8 for the VRAM headroom, not the speed.

### 5c. Prefill (sm_90a): the levers that actually moved, in landed order

From 446 ms to 176 ms @4k fp8 (each verified token-identical; full chain in the
campaign doc): w8a8 QGMMA + TMA staging; quant-into-norm and GLU-into-quant
fusions (RmsNorm t3/t4, QuantFp8 t3/t4 — fuse INTO the producer, quantize from
the rounded value); dedicated per-class kernel objects (the TU-isolation law);
uniform m128n256 tiles (the smem-wall escape); band rasterization; **the
384-thread producer/consumer GEMM object** (the largest single step, -32 ms —
justified by the cuBLASLt ceiling measurement); flash BKV 64/32; overhang
ladder rungs (-30 ms); CUDA-graph segment chains (~1 ms).

---

## Stage 6 — benchmark protocol (how not to quote a wrong number)

Drive BOTH engines with the same harness and verify they did the same work:

    vllm bench serve --backend openai-chat --base-url <url> \
      --endpoint /v1/chat/completions --model <served-name> --tokenizer <model> \
      --dataset-name random --random-input-len <I> --random-output-len <O> \
      --num-prompts <N> --max-concurrency <C> --seed 0

Mandatory checks:

- **Arms run sequentially and exclusively** — one server at a time; kill and
  drain between arms (`bringup_showdown.sh` is the template). Medians over >=5
  rounds (9 for a headline number). The first request after server start pays a
  one-time warmup (~57 ms observed) — medians absorb it, means do not.
- **Identical input tokens.** Compare `Total input tokens` across engines.
- **Identical output tokens.** `Total generated tokens` must match. plowrt must
  honor `ignore_eos` or it generates far fewer tokens per request (measured
  161 vs 512 -> 3.2x distortion).
- **Raise `--slo-ms`.** The default 250 ms sheds every request once predicted
  wait exceeds it, and the bench counts those 429s as **successful** — this
  produced a fake "2592 tok/s" reading. Always check `Total generated tokens`.
- **Sweep concurrency, do not quote one point.** plow's fixed-width decode makes
  its position strongly concurrency-dependent; on Gemma-4-12B it wins at c4,
  ties at c8, and loses at c16.
- **TTFT comparisons need `PLOW_TTFT_LOG=1` sanity once per config** — confirm
  the client TTFT decomposes into tokenize + prefill + ~0 (the GH200 CUDA arm
  does; if `UNACCOUNTED` is large, instrument before comparing).

---

## Canonical sm_90a prefill configuration (as of the GH200 campaign, T31-T37)

Build (all five objects):
```
PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32" \
PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_W8A8=1 PLOW_BUILD_SEG=1 \
PLOW_BUILD_FATLITE=1 PLOW_BUILD_GEMM_WS384=1 \
PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1 \
scripts/build_sm90a_cubin.sh <out-dir>/interp_sm90a.cubin
```
(Drop `PLOW_BUILD_W8A8` for bf16-only cubins; the ws384 lean object serves both
precisions.)

Emit (per bundle; add `PLOW_W8A8=1 PLOW_QNORM_FUSE=1` for fp8):
```
PLOW_NS_FULL_ABS=33 PLOW_TMA_GEMM=1 PLOW_MAX_CHUNK=8192 \
PLOW_PF_LADDER_APPEND=640,1152,2176,4224 PLOW_UNISEG_MAX_T=512 \
PLOW_NO_GLU_FUSE=1 PLOW_SEG_CLASS_SLICE=light PLOW_SEG_SLICE_ALL=1 \
PLOW_SEG_PURE_GEMM=<fp8|1> PLOW_SEG_FA512=all \
plowc --hf-dir <snapshot> --emit devblob --arch sm_90a --gpu h100 --max-ctx 8192 --out <bundle>
```

Serve:
```
PLOW_PF_SEG_DIR=<cubin-dir> PLOW_PF_SEG_PURE=<fp8|1> \
PLOW_PF_SEG_FA512=all PLOW_PF_SEG_GRAPH=1 plowrt serve --assets <bundle>
```

Emit knob and serve knob MUST pair (`PLOW_SEG_PURE_GEMM` ↔ `PLOW_PF_SEG_PURE`,
`PLOW_SEG_FA512` ↔ `PLOW_PF_SEG_FA512`) — the classing decides which object a
packet lands on, and a mismatched object `__trap()`s loudly (by design).

**Model generality (T37, proven on Gemma-4-31B unmodified):** the classing only
claims TMA-mapped plain GEMMs (unmapped falls to the fat object's cp.async
path) and only hd 256/512 flash (hd128 models keep flash on the fat object);
odd-N GEMMs trap loudly; the segment cap is 2048 (60-layer models emit ~603).
This stack supersedes the older note that segmented objects were "unreachable
from serve" — on sm_90a they ARE the shipping prefill path. On sm_120a,
`PLOW_UNISEG=1` remains the rule.

---

## Refuted ideas — do not retry without NEW evidence

All hardware-measured on GH200, correct implementations, kept in-tree behind
off-by-default knobs where noted:

- cluster-pair TMA multicast (`uni256_cluster_probe.cu`): 0 mismatches, 16-37%
  SLOWER — the rank-0 issuer serializes B for two CTAs; L2 service was never
  the wall.
- smem-staged coalesced epilogue: +15% standalone, -3 ms in-model (probe law).
- 2-deep wgmma window with a shared issuer: deadlocks (the issuer blocks on an
  arrival only its own warpgroup's next stage can make); at reduced issue lead
  it runs but loses 4%. With a DEDICATED producer it is safe but neutral.
- split dual-n128 acc chains: -7% (hardware pipelines the single n256 chain).
- warpgroup-per-item flash (`PLOW_NV_FA_WGITEM`): token-identical, neutral at
  equal BKV — the BKV=64 win was staging granularity, not barriers.
- bf16 256-byte k-stages at NS=2: -44 ms (ring starvation).
- quant on the ws384 consumers (`PLOW_SEG_V2=q8`): saves 2 launches/layer,
  halves quant's block parallelism: +8.4 ms.
- flash head-major work order, eq-smem launches, non-cooperative launches,
  per-entry slice halving, NS/band sweeps: all neutral.
- sm90a M=1 lm_head GEMV arm: corrupt first token AND zero perf delta (open bug).
- classing v2 rope/merge→FA on 256-thread objects: light bodies lose more at
  occ-1 than the merged launches save.
- (sm_120a) row-blocking GEMV variants that win 1.4x in isolation lose in the
  megakernel — the original probe-law instance.

---

## Can knobs be tuned without running the whole network?

Partly — and the split matters, because getting it wrong produces confident
wrong answers.

**Isolated kernel microbenches are PRUNERS, not scorers.** The tree ships many
(`runtime/tests/gemv_batch_sm120.cu`, `batch_decode_sm120.cu`,
`flashdec_fp8_bw_sm120.cu`, `runtime/nvidia/experiments/gemv_lab_h100.cu`,
`uni256_probe.cu`, `uni256_cluster_probe.cu`, `tma_uni_gemm_ab.cu`, …) and they
build and run standalone in seconds:

    nvcc -std=c++17 -O3 -arch=sm_120a -Iinclude -Iruntime/common -Iruntime/nvidia \
      runtime/tests/gemv_batch_sm120.cu -o /tmp/gemvbatch -lcuda

They are excellent for **correctness gating a rung** (gemv_batch caught
unwritten rows at M>1; uni256_cluster proved the multicast protocol) and
**shape-level scanning**. They do not predict the megakernel — see the probe
law. Two measured instances: `GV_MM_MAX=16` scored 1.00x isolated and 1.43x
in-model (sm_120a); the smem-staged epilogue scored +15% isolated and -3 ms
in-model (sm_90a).

**The middle tier — a single transformer block — is the one to actually sweep
on.** `plowc --block l` / `--block l..r` emits a block asset and
`examples/block_run` drives it through the REAL `GpuEngine` and the REAL
megakernel:

    plowc --hf-dir <model> --gpu <gpu> --emit devblob --block 0..2 --out <dir>
    block_run <dir> bench --batch 8,16 --ctx 1024 --iters 60 --warmup 10
    block_run <dir> check [--in x.npy]          # shape / finiteness gate

It reproduces full-model ratios within ~1.4% (the GV_MM_MAX case: block_run
1.45x vs full model 1.43x, where the isolated bench said 1.00x) at 10-20x lower
cost per config. Caveat: `--batch` selects active slots, not kernel width —
emit one block asset per decode batch you want to score. Layer kinds are not
interchangeable: emit one block per kind and score the kind-weighted sum of
MARGINAL per-layer cost (`score = N_slide*L_slide + N_full*L_full`); the fixed
per-block overhead and the lm_head cancel in a ranking.

**The final scorer is `step_bench`** — real blob, real megakernel, no HTTP/mux:

    cargo build --release -p plowrt --features cuda --example step_bench
    PLOW_STEP_TIME=1 ./target/release/examples/step_bench <assets> [slots] [ctx] [steps]

The practical loop, cheapest first:

| tier | harness | cost/config | use it for |
|------|---------|-------------|------------|
| 1 prune | `runtime/tests/*.cu`, `experiments/*.cu` | seconds, no blob | correctness gating a rung; killing obviously-bad shapes. **Never trust its ranking.** |
| 2 sweep | `plowc --block` + `block_run bench` | ~3 s | the wide sweep — real megakernel, reproduces full-model ratios |
| 3 confirm | `step_bench` (whole blob, no HTTP/mux) | ~1 min | final scoring on the real layer mix |
| 4 accept | `vllm bench serve` vs the engine | minutes | end-to-end, and the only tier that sees host-gap knobs |

Tier 2 is automated by `scripts/tune_block_sweep.sh`; `scripts/tune_decode_sweep.sh`
covers tier 1 -> 3.

### What tier 2 cannot score

- **Segmented prefill dispatch knobs** (classing, per-object block size, seg
  graphs): block_run drives the decode/uniseg path; the sm_90a segmented
  prefill stack is only exercised by tier 4 (serve + client).
- **Prefill bucket policy** (`PLOW_PF_COVER`, `PLOW_PF_CHUNK_COST`,
  `PLOW_PF_LADDER_APPEND`): needs varied prompt lengths, i.e. tier 4.
- **Host-gap knobs** (`PLOW_DEV_SAMPLE`, `PLOW_MULTISTEP`, `--slo-ms`) — their
  effect is between kernels; no single-engine harness sees it.
- `PLOW_MAX_CHUNK` needs no benchmark at all: it is an analytic memory-sizing
  decision (the ring formula), derived from the model's window automatically.

## Tuner status and honest impact

`plowc tune --gpu <gpu> --status` reports the cell and whether measurements
exist; `plowc tune` itself is query-only (populating a cell needs
`scripts/tune_decode_sweep.sh`).

**On NVIDIA targets the tunedb currently changes NOTHING (verified T37):**
`pick_tile` short-circuits to one canonical opcode per precision
(`nvidia_prefill_gemm_op`) — tile geometry is fixed by cubin `-D` macros — and
`plowc tune gemm` is a gfx950-only campaign (builds via hipcc, measures the AMD
object). The CompilerOracle/tunedb machinery prices AMD tiles only. If
per-shape choices ever matter on NVIDIA (n128-vs-n256 by (M,N,K), BKV per head
dim, ws384-vs-uniform), the oracle is the right home — but today those are
build/env knobs and nothing on the NVIDIA path consults measurements.

**Measured tuner-axis impact is small everywhere.** Hand-driving the two axes
the decode tuner sweeps gave ~4.3% (`NS_FULL_ABS`) + ~2% (`GV_UN16`). The large
wins on both campaigns were *structural*: `GV_MM_MAX` (+43% kernel),
`DEV_SAMPLE`+`MULTISTEP` (1.74x), `PLOW_MAX_CHUNK` (5x KV) on sm_120a; the
kernel-object architecture, the 384-thread GEMM, and the overhang ladder on
sm_90a. Budget effort accordingly: **fix the structure and the defaults first,
tune last.**

---

## Scripts in this directory

- `bringup_gate.sh <assets> <tag> <port> [plowrt]` — token-identity gate.
- `bringup_bench.sh <tag> <url> <model> <tokenizer> [round]` — one client pass;
  env `IN_LENS`, `NPROMPT`, `OUTLEN`, `BRINGUP_OUT`. Appends to `cells.tsv`.
- `bringup_showdown.sh` — sequential-exclusive multi-arm template (edit the
  arm list at the bottom).
- `bringup_ceiling.py` — cuBLASLt/torch fp8+bf16 ceilings at your model's
  exact GEMM shapes (edit the shape list).
- `gpulease <label> <cmd>` — contention-audited GPU run wrapper (rule 1).
- `quantize_fp8.py <src> <out>` — fp8 weight twins for w8a8 bundles.

All bring-up output lands under `BRINGUP_OUT` (default `/tmp/bringup-$USER`);
nothing here writes into the repo.
