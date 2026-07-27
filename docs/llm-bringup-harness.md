# LLM bring-up harness

Order of operations for bringing a new model up on plow and getting it to a
defensible performance number. Derived from the Gemma-4-12B / RTX 5090 (sm_120a)
campaign; the measurements behind every claim are in
`perf-data/gemma4-12b-sm120-serving.md`.

The ordering is the point. **The interpreter must be optimal and every kernel
arm the geometry needs must be present in the cubin BEFORE any blob is emitted
or any number is quoted.** A missing arm is not a crash — it is a silently
wrong answer or a silent fallback, and every stage below is designed to fail
loudly instead.

---

## Stage 0 — environment sanity (do this first, it invalidates everything else)

| check | command | why |
|-------|---------|-----|
| driver vs libcuda | `nvidia-smi --query-gpu=driver_version --format=csv` | a CUDA **compat** libcuda NEWER than the driver fails `cuInit` |
| GPU actually used | grep the serve log for `backend ready — GPU accelerated` | CPU fallback is a **WARNING**, not an error |
| toolchain | `nix develop` | cargo/rustc are not on the default PATH |

    # If cuInit fails with CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
    export PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1

**Trap.** Without `PLOW_LIBCUDA` pointing at the system lib, plowrt drops to the
CPU reference interpreter and still serves correct text — slowly. An unwary
benchmark measures the CPU path. Always confirm the `GPU accelerated` line.

---

## Stage 1 — geometry audit (decides which kernels you need)

Read `config.json` and write these down. They select kernel arms and set the
VRAM budget; everything downstream depends on them.

| field | why it matters |
|-------|----------------|
| `layer_types` (full vs sliding count) | selects flash-decode arms; sliding layers get a ring |
| `head_dim` per family | heterogeneous hd (e.g. 256 sliding / 512 full) forces per-HD template instantiations |
| `num_key_value_heads` per family | kvh=1 full layers are a different GEMV shape from kvh=8 |
| `sliding_window` | sets the KV ring floor — see the budget formula |
| `vocab_size` | lm_head GEMV + sampler scratch |
| `rope_parameters` sub-objects | per-family RoPE recipes (v7 blob) |
| MoE vs dense | entirely different op family (`op_moe.cuh`) |

Gemma-4-12B for reference: 48 layers = 8 full (kvh 1, hd 512) + 40 sliding
(kvh 8, hd 256), window 1024, vocab 262144, dense, bf16.

---

## Stage 2 — interpreter + kernel presence (the gate)

Build the cubins, then **prove** the arms exist. Do not skip the verification —
this is the stage that catches a geometry the objects were never compiled for.

    PLOW_ROOT=$PWD scripts/build_sm120_cubin.sh <assets>/interp_sm120.cubin

Then verify:

    # 1. entry symbols present (the script already fails hard if not)
    cuobjdump -symbols <assets>/interp_sm120.cubin    | grep _Z12interp_sm12011PlowProgram
    cuobjdump -symbols <assets>/interp_sm120_pf.cubin | grep _Z15interp_sm120_pf11PlowProgram
    cuobjdump -symbols <assets>/sample_sm120.cubin    | grep -E 'plow_sample|plow_advance'

    # 2. register pressure / spill of the megakernel
    cuobjdump -res-usage <assets>/interp_sm120.cubin | grep -A1 interp_sm120

`plow_advance` in the sampler object is what gates `PLOW_MULTISTEP`; if it is
missing, multi-step silently disables itself with a warning and you lose ~1.7x.

**Register reading.** The interpreter is ONE megakernel, so its register count
is the MAX over every instantiated arm — a wide rung added for batching taxes
the B=1 path too. On sm_120a the observed decode object sits at REG:250 with a
1024 B stack frame. Counterintuitively, *more* spill measured faster here (see
the `GV_UN16` sweep): this rung is latency-bound, not register-bound. Do not
assume spill == slow; measure.

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
| `GV_MM_MAX=16` | 8 | **set when the blob's decode batch >= 16** — see below |
| `GV_UN16=8` | 4 | wide-rung unroll; best measured on sm_120a |

### Full cubin knob inventory

Every compile-time knob the sm_120 objects read, with its in-source default.
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

**Ablation / experiment switches — default OFF, leave off:**
`PLOW_NV_GEMV_RB`, `PLOW_NV_RB_GEMV`, `PLOW_NV_RB_QKV`, `PLOW_NV_RB_LMHEAD`,
`GV_RB`, `GV_UNROLL_RB`, `GV_LS_SG`, `PLOW_NV_GEMV_LS`,
`PLOW_NV_GEMV_NOSTAGE`, `PLOW_NV_GEMV_STAGE_MINROWS`, `PLOW_NV_FP8_RB`,
`PLOW_NV_FA_TMA`, `PLOW_NV_FA_WPR`, `PLOW_NV_FA_WPR_RB`, `PLOW_NV_FA_VDBUF`,
`PLOW_NV_FA_QGLOB`, `PLOW_NV_FA_CORRSKIP`, `PLOW_NV_FA_REDBOUND`,
`PLOW_NV_FA_FP8ABL`, `PLOW_NV_KVBOUNDS`, `PLOW_NV_LEAN_DECODE`,
`PLOW_NV_SEGMENTS`, `PLOW_NV_SEG_GEMM`, `PLOW_NV_SKELETON`,
`PLOW_NV_SKEL_PAD`, `PLOW_NV_TRACE`, `PLOW_NV_ABLATE_LO/HI`.

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
stays at 8192 when `window == 0`: at 8192 a window-1024
model rings 16384 rows, an 8x inflation over what the window needs, purely
because of the chunk. Measured on Gemma-4-12B at B=16/ctx 8192, the old default
wanted 21.32 GiB of KV against 6.09 GiB now — it did not fit beside 12 GiB of
weights on a 32 GiB card. There is a hard floor at `next_pow2(window)`, so
window 1024 cannot go below 2048 rows however small the chunk.

Emit knobs:

    # NS_FULL_ABS and MAX_CHUNK now default correctly — only these are needed:
    PLOW_UNISEG=1 PLOW_DECODE_BATCH=<B> \
    [PLOW_FP8_KV=1] [PLOW_FP8=1 PLOW_W8A8=1] \
      plowc --hf-dir <model> --gpu <gpu> --emit devblob --max-ctx <ctx> \
            [--weight-dtype fp8] --out <dir>          # <dir>, NOT <dir>/model.pkt

`--out` pointing at a `.pkt` path emits a bare blob (`bundle=false`); point it
at a DIRECTORY to get a servable bundle (checkpoint symlink, tokenizer,
weights.json).

**`PLOW_DECODE_BATCH` is a fixed kernel width, not a maximum.** A B=32 blob pays
the full B=32 step even when 16 slots are live — measured 165 tok/s at c16 vs
240 for a B=16 blob. Match B to expected concurrency.

### fp8 weights need a pre-quantized checkpoint

plow does not quantize on load; the runtime looks for `fp8/<name>` twins and
fails with `MISSING WEIGHT: fp8/...` if absent.

    python3 perf-data/harness/quantize_fp8.py <src-model-dir> <out-dir>
    # then a checkpoint dir holding BOTH (the loader globs *.safetensors):
    #   model.safetensors      -> original bf16 (norms/embeddings/lm_head)
    #   model-fp8.safetensors  -> the fp8 twins

---

## Stage 4 — correctness gate (before ANY performance number)

Run every one of these. They are cheap and each catches a distinct silent-wrong.

1. **Greedy sanity** — a short factual prompt at `temperature:0`. Garbage here
   means a wrong cubin/blob pairing, not a tuning problem.
2. **Multi-chunk prefill** — a prompt longer than `PLOW_MAX_CHUNK` to exercise
   ring wraparound on sliding layers.
3. **Stop behavior** — confirm generation stops at eos, and that
   `"ignore_eos": true` runs to `max_completion_tokens` with
   `finish_reason: "length"`.
4. **Cap honored** — send `max_completion_tokens` and confirm the response
   respects it (OpenAI renamed `max_tokens`; a server binding only the old name
   silently runs to eos).
5. **Lossy modes are opt-in** — fp8 KV diverges from greedy bf16 after ~21
   tokens by design. Never compare a lossy plow config against a bf16 baseline
   without saying so.

---

## Stage 5 — performance bring-up ladder

Turn on `PLOW_STEP_TIME=1` and read one line:

    step_slots means ... gap_us=<G> dev_interp_ms=<K> dev_upload_us dev_download_us

This splits the step into **device kernel (K)** and **host gap (G)**. Follow the
branch:

    if G >> K:  host-bound   -> Stage 5a
    if K >> G:  kernel-bound -> Stage 5b
    if K ~ G:   both; do 5a first (it is free)

### 5a. Host-bound

Device sampling and multi-step are now **ON by default** (`PLOW_DEV_SAMPLE=0` /
`PLOW_MULTISTEP=0` opt out); confirm both lines appear in the serve log. Add:

    PLOW_PF_BATCH=1       # cross-request batched prefill (still opt-in)

Measured on Gemma-4-12B at B=16: gap 83.6 -> 32.2 ms, throughput 102.74 ->
185.60 tok/s. **This was the single largest free win, which is why it is now the
default.** K=8 captures nearly all of it; K=32 adds ~3%.

`PLOW_PF_INTERLEAVE` / `PLOW_PF_CHUNK` measured at or below their defaults on
this workload — leave them alone unless the gap is provably prefill.

### 5b. Kernel-bound

Ladder, biggest first:

1. **`GV_MM_MAX`** — the decode GEMV ladder is `{1,2,4,8}` and walks M in blocks
   of 8 above that, so **B=16 streams the entire weight set twice**. Measured
   B=16 = 1.85x B=8 for identical weights. `-DGV_MM_MAX=16` took the B=16 kernel
   41.17 -> 28.8 ms. With the default ladder, every 8 slots costs one full
   weight read, capping aggregate decode near 437-465 tok/s at any batch.
   Note it *hurts* B=8 (355 -> 294 tok/s) — set it only for B>=16.
2. **`PLOW_NS_FULL_ABS`** — decode split factor. Monotone at B=16 on 170 SMs:
   8 -> 25.18 ms, 16 -> 25.28, 32 -> 25.93, 48 -> 26.32. The value the family
   build script documents (48) is the worst; 16x48=768 blocks oversubscribes.
3. **`GV_UN16` / `GV_UN_GLU16`** — wide-rung unroll, ~2%.
4. **Weight dtype** — fp8 weights cut 22.2 -> 12.0 GiB but only 28.8 -> 26.5 ms,
   because that GEMV is dequant-**compute**-bound (1046 GB/s), not
   bandwidth-bound. Take fp8 for the VRAM headroom, not the speed.

---

## Stage 6 — benchmark protocol (how not to quote a wrong number)

Drive BOTH engines with the same harness and verify they did the same work:

    vllm bench serve --backend openai-chat --base-url <url> \
      --endpoint /v1/chat/completions --model <served-name> --tokenizer <model> \
      --dataset-name random --random-input-len <I> --random-output-len <O> \
      --num-prompts <N> --max-concurrency <C> --seed 0

Mandatory checks:

- **Identical input tokens.** Compare `Total input tokens` across engines.
- **Identical output tokens.** `Total generated tokens` must match. plowrt must
  honor `ignore_eos` or it generates far fewer tokens per request and pays
  extra prefill churn per output token (measured 161 vs 512 -> 3.2x).
- **Raise `--slo-ms`.** The default 250 ms sheds every request once predicted
  wait exceeds it, which any large batch does. The bench counts those 429s as
  **successful** requests — this produced a fake "2592 tok/s" reading. Always
  check `Total generated tokens` is what you asked for.
- **Sweep concurrency, do not quote one point.** plow's fixed-width decode makes
  its position strongly concurrency-dependent; on Gemma-4-12B it wins at c4,
  ties at c8, and loses at c16.
- **TTFT from plowrt is not comparable** — it emits the role chunk before
  compute, so its TTFT is an SSE artifact, not a prefill measurement.

---

## Knob reference by layer

### Cubin (nvcc `-D`, via `PLOW_EXTRA_DEFINES`)
| knob | default | recommended |
|------|---------|-------------|
| `GV_MM_MAX` | 8 | **16 when decode batch >= 16** |
| `GV_UN16` / `GV_UN_GLU16` | 4 / 2 | 8 / 2 on sm_120a (~2%) |
| `PLOW_FP8_KV` | off | on when batch x ctx does not fit in bf16 |
| `PLOW_NV_W8A8` + `PLOW_NV_FA_PIPE=0` | off | with fp8 weights |

### plowc (emit)
| knob | default | recommended |
|------|---------|-------------|
| `PLOW_UNISEG` | off | **=1 mandatory on sm_120** |
| `PLOW_NS_FULL_ABS` | **8** (family script) | leave; override for a sweep |
| `PLOW_MAX_CHUNK` | **derived from window** (1024 for Gemma-4; 8192 all-global) | leave; =8192 for long-prompt-only with VRAM to spare |
| `PLOW_DECODE_BATCH` | 1 | match expected concurrency (fixed width) |
| `PLOW_FP8` / `PLOW_W8A8` | off | for VRAM headroom; needs fp8 twins |
| `--max-ctx` | 131072 | the smallest ctx you will actually serve |

### plowrt (serve)
| knob | default | recommended |
|------|---------|-------------|
| `PLOW_DEV_SAMPLE` | **on** (=0 opts out) | leave on |
| `PLOW_MULTISTEP` | **on, K=8** (=0 opts out) | =32 for max throughput, at coarser streaming |
| `PLOW_PF_BATCH` | off | =1 |
| `--slo-ms` | 250 **floor**, effective `max(250, 8 x service_ms)` | leave; it now scales with batch |
| `PLOW_STEP_TIME` | off | =1 while tuning only |
| `PLOW_LIBCUDA` | — | set if the compat libcuda is newer than the driver |

**These defaults are now flipped in-tree** (`PLOW_DEV_SAMPLE`, `PLOW_MULTISTEP`,
`PLOW_MAX_CHUNK`, `PLOW_NS_FULL_ABS`, batch-scaled `--slo-ms`). A B=16
Gemma-4-12B bundle with NO tuning env vars and no `--slo-ms` now serves 525.55
tok/s at c16 with zero shed events; the same bundle before the flips gave
102.74 tok/s and 429'd every request. `GV_MM_MAX` is deliberately NOT flipped:
16 wins at B>=16 but LOSES at B=8 (355 -> 294 tok/s) and the default blob is
B=1, so it stays a per-build `PLOW_EXTRA_DEFINES` choice.

---

## Can these be tuned without running the whole network?

Partly — and the split matters, because getting it wrong produces confident
wrong answers.

**Isolated kernel microbenches are PRUNERS, not scorers.** The tree ships a lot
of them (`runtime/tests/gemv_batch_sm120.cu`, `batch_decode_sm120.cu`,
`flashdec_fp8_bw_sm120.cu`, `runtime/nvidia/experiments/gemv_lab_h100.cu`,
`fp8_gemv.cu`, `fa_gf_full_ab.cu`, …) and they build and run standalone in
seconds:

    nvcc -std=c++17 -O3 -arch=sm_120a -Iinclude -Iruntime/common -Iruntime/nvidia \
      runtime/tests/gemv_batch_sm120.cu -o /tmp/gemvbatch -lcuda

`gemv_batch_sm120.cu` already carries Gemma-4-12B decode shapes and sweeps
MM=1/8/16/32 with HBM-resident weights and an all-rows correctness check. It is
excellent for what it is for: **correctness gating a rung** (it caught that
`d_gemv` at M>1 left rows unwritten) and **shape-level scanning**.

But it does not predict the megakernel. `scripts/tune_decode_sweep.sh` says so
outright — `gemv_lab_h100.cu` measures row-blocking winning 1.4x on every decode
shape, and in context it loses — and this campaign produced a second instance:
the harness times one `gemv_rows<16>` as costing the SAME as walking two
`gemv_rows<8>` (0.163 vs 0.163 ms on `q_proj [4096,3840]`), while in the
megakernel `-DGV_MM_MAX=16` was worth 41.17 -> 28.8 ms. The isolated kernel gets
full occupancy and its own register budget; the interpreter runs one block per
SM with a register ceiling shared across every instantiated arm.

**The middle tier — a single transformer block — is the one to actually sweep
on.** `plowc --block l` / `--block l..r` emits a block asset (blob +
`block.json` descriptor) and `examples/block_run` drives it through the REAL
`GpuEngine` and the REAL megakernel, so it keeps the context the microbench
loses, at a fraction of the cost:

    plowc --hf-dir <model> --gpu <gpu> --emit devblob --block 0..2 --out <dir>
    block_run <dir> bench --batch 8,16 --ctx 1024 --iters 60 --warmup 10
    block_run <dir> check [--in x.npy]          # shape / finiteness gate

**It reproduces the full-model result.** Same two cubins, Gemma-4-12B, B=16:

| harness                    | GV_MM_MAX=8 | =16       | ratio |
|----------------------------|-------------|-----------|-------|
| isolated GEMV microbench   | 0.163 ms    | 0.163 ms  | 1.00x (WRONG) |
| **block_run (2 layers)**   | 1564.92 us  | 1076.34 us| **1.45x** |
| full model (48 layers)     | 41.17 ms    | 28.8 ms   | 1.43x |

Within 1.4% of the full model, where the isolated kernel said "no difference".
It also predicts absolute cost: 1076 us x 48/2 = 25.8 ms vs 25.6 ms measured
end-to-end.

And it is cheap: 2.7 GiB of weights instead of 22.2, one `block_run` invocation
(load + 60-iter sweep) is **3.3 s wall** against a full-model config that needs
3.3 s of weight upload alone plus serve bringup plus the benchmark — call it
10-20x per config, which is what makes a wide sweep affordable.

Caveat: `--batch` selects active slots, not kernel width. A block emitted with
`PLOW_DECODE_BATCH=16` runs the B=16 kernel whatever `--batch` says (visible
above: 1515 us at "B=8" vs 1565 us at B=16). Emit one block asset per decode
batch you want to score.

**The final scorer is `step_bench`, and it too does NOT need the network.** It drives
`GpuEngine` directly — real blob, real megakernel, no HTTP, no mux, no dataset:

    cargo build --release -p plowrt --features cuda --example step_bench
    PLOW_STEP_TIME=1 ./target/release/examples/step_bench <assets> [slots] [ctx] [steps]

So the practical loop is three tiers, cheapest first:

| tier | harness | cost/config | use it for |
|------|---------|-------------|------------|
| 1 prune | `runtime/tests/*.cu`, `experiments/*.cu` | seconds, no blob | correctness gating a rung; killing obviously-bad shapes. **Never trust its ranking.** |
| 2 sweep | `plowc --block` + `block_run bench` | ~3 s | the wide sweep — real megakernel, reproduces full-model ratios |
| 3 confirm | `step_bench` (whole blob, no HTTP/mux) | ~1 min | final scoring on the real layer mix |
| 4 accept | `vllm bench serve` vs the engine | minutes | end-to-end, and the only tier that sees host-gap knobs |

Tier 2 is automated by `scripts/tune_block_sweep.sh` (matrix file ->
parallel nvcc -> serial block_run -> ranked TSV). `scripts/tune_decode_sweep.sh`
covers tier 1 -> 3.

### Scoring a model from blocks

Layer kinds are not interchangeable — Gemma-4 is 40 sliding (hd 256, kvh 8) + 8
full (hd 512, kvh 1) — and a sliding-only block cannot score `PLOW_NS_FULL_ABS`
at all, since the emitter filters it on `gemv_family && full`. Emit one block
per kind and score the kind-weighted sum of MARGINAL per-layer cost:

    score = N_slide * L_slide + N_full * L_full

Measured on Gemma-4-12B at B=16: 1 sliding layer 545.46 us, 2 sliding layers
1075.84 us => L_slide = 530.38 us and a fixed per-block overhead O = 15.08 us
(the block declares embedding/lm_head weights). 1 full layer 604.41 us =>
L_full = 589.33 us. Score 25.94 ms vs 28.8 ms measured full-model; the ~2.9 ms
residual is the lm_head GEMV (~1.6 ms bf16 at 262144 vocab) plus embed/final
norm, which the block does not run. Both O and the head are CONSTANT across
knobs, so they cancel in a ranking — the score is a comparator, not a TPOT.

### What tier 2 cannot score

- **Segmented dispatch** (`PLOW_NV_SEGMENTS`, `PLOW_NV_SEG_GEMM` — the
  "switch interpreter / wave size" objects). Not a tier-2 limitation but a
  runtime one: the serve path requires a single coarse segment
  (`check_coarse_single_segment`), and a segmented bucket **disables prefill
  outright** (`exec/gpu.rs`: "Absent cubin, a segmented bucket, or a missing GQ
  appendix disables prefill"). `PLOW_UNISEG=1` is mandatory on sm_120 for this
  reason. Those objects are unreachable from serve today, so every measurement
  in this file is a uniseg measurement, and scoring them would measure a path
  nothing runs. Making them reachable is runtime work, not a tuning knob.
- **Prefill bucket policy** (`PLOW_PF_COVER`, `PLOW_PF_CHUNK_COST`). These ARE
  live — `pick_prefill_bucket` minimises padded rows against a per-launch cost —
  but `block_run bench` runs a FIXED `--ctx`, so the bucket pick is trivial and
  all four settings measured within 0.2% (26549-26598 us). Scoring them needs
  varied prompt lengths, i.e. tier 4.
- **Host-gap knobs** (`PLOW_DEV_SAMPLE`, `PLOW_MULTISTEP`, `--slo-ms`) — their
  effect is between kernels; no single-engine harness sees it.

### Sweep results so far (Gemma-4-12B, sm_120a, B=16, ctx 1024)

16 configs over two sweeps: `PLOW_NV_FA_GF` {2,4,8}, `PLOW_NV_FA_GF_FULL`
{2,4,8}, `GV_UNROLL` {4,8,16}, `GV_UNROLL_GLU` {4,8}, `GV_UN16` {4,8},
`PLOW_NV_GATE_SLEEP` {16,64,256}, `PLOW_NV_FA_KUN` {1,2}, and the bucket-policy
env knobs. **The committed baseline won every one.** Spread was under 0.6%
except `GV_UN16=8` (+4.4%) and `PLOW_NV_FA_GF_FULL=8` (+0.5%). Note `GV_UN16=8`
measured BEST on the full model with fp8 weights (25.04 vs 25.61 ms) and worst
here with bf16 weights — that knob is precision-dependent, so sweep it at the
precision you ship.

**What cannot be tuned this way at all:** `PLOW_DEV_SAMPLE`, `PLOW_MULTISTEP`
and `--slo-ms` are host/device *interaction* knobs — their whole effect is the
gap between kernels, which no kernel benchmark and no single-engine harness can
see. Those need the serving loop. `PLOW_MAX_CHUNK` needs no benchmark at all: it
is an analytic memory-sizing decision (see the ring formula), now derived from
the model's window automatically.

## Tuner status and honest impact

`plowc tune --gpu <gpu> --status` reports the cell and whether measurements
exist. With no entry it prints *"no kernel measurements for this cell — selection
will use the analytical model and report tier `portable`"*, which is what
sm_120a does today (`tuning/` holds only `nvidia/sm_90a/h100-nvl`).

`plowc tune` is **query-only**. Populating a cell needs
`scripts/tune_decode_sweep.sh`, which currently requires `gpulease` and forces
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`; on a box whose driver is older
than that compat lib the sweep cannot `cuInit`. Its occupancy defaults are also
H100-shaped (`1:132 2:264`) and need `--occ "1:<sm_count>"`.

**Measured tuner-axis impact is small.** Hand-driving the two axes the decode
tuner sweeps gave ~4.3% (`NS_FULL_ABS`) and ~2% (`GV_UN16`) — together ~6%. The
large wins were *structural defaults*, not tuning: `GV_MM_MAX` (+43% kernel),
`DEV_SAMPLE`+`MULTISTEP` (1.74x), `PLOW_MAX_CHUNK` (5x KV). Budget effort
accordingly: fix the defaults first, tune last.
