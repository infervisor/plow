# Qwen ASR playbook — adding models and beating vLLM

Guide for the `shaswot/qwen-asr` optimization campaign. Read alongside:

- `docs/agent1-repository-map.md` — architecture map
- `docs/agent2-benchmark-contract.md` — immutable benchmark rules
- `docs/final-validation.md` — current PASS/FAIL gate vs vLLM

---

## Part 1 — Adding a new model

### 1.1 What plow supports today

The production GPU path is **`devgen`** (via `plowc --hf-dir`), not `nn-graph`
analysis or egglog rewrite. Supported HuggingFace `model_type` values in
`crates/devgen/src/config.rs`:

| `model_type` | plow arch | notes |
|---|---|---|
| `qwen3` | Qwen3 | GQA, q/k RMSNorm, SwiGLU, tied lm_head — **Qwen3-4B campaign model** |
| `llama` | Llama | standard dense decoder |
| `gemma4_text` | Gemma4 | flat text-only export |
| Gemma-4 multimodal | Gemma4 | reads `text_config` subtree |

**Not supported:** MoE-only paths without dense fallback, MLA/GLM/Nemotron (separate
emitters), audio/ASR encoders, unknown `model_type`. Adding a new family requires
`devgen` + kernel work, not just a checkpoint drop.

**Naming note:** “Qwen ASR” in this branch means **Qwen3 dense text decoder**
optimization, not a speech model. There is no Qwen-Audio stack in this tree.

### 1.2 Prerequisites

| item | requirement |
|---|---|
| Environment | `nix develop` (compiler, nvcc/hipcc, cargo) |
| Weights | local HF dir with `config.json`, `*.safetensors`, `tokenizer.json` |
| GPU | same ISA you compile for (RTX 5090 → `sm_120a`) |
| Driver | CUDA (`libcuda`) or ROCm (`libamdhip64`) — CPU fallback is correctness-only |

### 1.3 Step-by-step: compile assets for serve

**1. Enter the dev shell**

```bash
cd /path/to/plow
nix develop
```

**2. Emit the device blob + manifest**

```bash
CKPT=/path/to/MyModel          # HF checkpoint dir
OUT=/path/to/assets/my-model-sm120

plowc --hf-dir "$CKPT" \
  --arch sm_120a \
  --max-ctx 131072 \
  --emit devblob \
  --out "$OUT"
```

This writes:

| file | purpose |
|---|---|
| `model.pkt` | PLOWDEV program the GPU interpreter runs |
| `weights.json` | tensor name → checkpoint shard map |
| `build.json` | manifest: opcodes, head_dim, prefill buckets, arch tag |

Or emit blob **and** cubin in one step (needs CUDA toolkit):

```bash
plowc --hf-dir "$CKPT" --arch sm_120a --emit devblob+cubin --out "$OUT"
```

**3. Build interpreter cubins (if not using `devblob+cubin`)**

NVIDIA sm_120 (RTX 5090):

```bash
# Preferred: CMake target (see runtime/CMakeLists.txt)
cmake -S runtime -B build-sm120 -DPLOW_SM120_CUBIN=ON
cmake --build build-sm120 --target sm120_cubins
cp build-sm120/cubin/interp_sm120.cubin       "$OUT/"
cp build-sm120/cubin/interp_sm120_pf.cubin    "$OUT/"   # prefill object — required for TTFT

# Legacy wrapper (deprecated):
# scripts/build_sm120_cubin.sh "$OUT/interp_sm120.cubin"
```

AMD gfx950:

```bash
scripts/build_gfx950_qwen.sh   # Qwen hd=128 flash prefill inline
# or the gfx950 CMake path documented in docs/amd/
```

Copy into the assets dir:

- decode: `interp_sm120.cubin` (or profile-specific name from `build.json`)
- prefill: `interp_sm120_pf.cubin` (must match manifest prefill buckets + head_dim)

**4. Link checkpoint**

```bash
ln -s "$CKPT" "$OUT/checkpoint"
# checkpoint/ must contain the same safetensors weights.json references
```

**5. Copy tokenizer**

```bash
cp "$CKPT/tokenizer.json" "$OUT/"
```

**6. Verify the layout**

A servable assets dir looks like:

```
my-model-sm120/
  model.pkt
  weights.json
  build.json
  tokenizer.json
  interp_sm120.cubin
  interp_sm120_pf.cubin      # omit only if you accept decode-loop prefill (slow TTFT)
  checkpoint/                → symlink to HF weights
```

**7. Smoke test**

```bash
cargo build -p plowrt --release
PLOW_GPU_TEST=1 PLOW_GPU_ASSETS="$OUT" cargo test -p plowrt -- gpu_lifecycle -- --nocapture

plowrt serve --assets "$OUT" --port 8101
# curl /v1/chat/completions or use vllm bench serve --backend openai-chat
```

Refuse to benchmark if logs show `CPU reference backend active`.

### 1.4 Adding a genuinely new architecture

If `model_type` is not in the table above:

1. **Parse config** — add a `cfg_*` in `crates/devgen/src/config.rs`
2. **Emit graph** — extend `devgen::emit_phase` for layer topology (norm, MLP, attn)
3. **Kernels** — ensure `runtime/nvidia/interp_sm120.cu` dispatches every opcode +
   head_dim your model needs (prefill traps if missing — see Qwen hd=128 history)
4. **Manifest** — `crates/devgen/src/manifest.rs` records shapes for cubin build
5. **Tests** — `runtime/tests/sm120_interp_op_test.cu` oracle for new attention shapes
6. **Chat template** — if serving via `/v1/chat/completions`, add a probe arm in
   `serve/chat.rs` (`gpu_chat_prompt`) so token counts match vLLM

Expect days–weeks of work for a new family; Qwen3/Llama/Gemma reuse most of the stack.

### 1.5 Common pitfalls

| pitfall | fix |
|---|---|
| Prefill cubin missing or wrong hd | TTFT uses decode-loop (ms × L per token) — build `_pf` with correct `PLOW_DOP_FLASH_PREFILL` |
| Cubin SM mismatch | re-build for `--arch` recorded in `build.json` |
| Gemma chat template on Qwen | add ChatML probe; wrong template ≠ 27 s TTFT but breaks fair vLLM comparison |
| Weights name mismatch | re-emit with `--hf-dir` pointing at the exact checkpoint |
| `head_dim` ≠ hidden/heads | Qwen3 uses explicit hd=128; do not infer from hidden/heads |

---

## Part 2 — Suggestions to beat vLLM

Current gate (`docs/final-validation.md`, RTX 5090, Qwen3-4B bf16):

| metric | gap vs vLLM |
|---|---|
| TTFT L=1024 | **+75%** (104 ms vs 59 ms) |
| TTFT L=4096/8192 | **+56%** |
| TPOT / ITL | **+3–7%** |
| Throughput | vLLM faster |

Optimizations on this branch already delivered **65–83× TTFT improvement** vs
baseline decode-loop prefill. The remaining gap is **not** host sync — it is
kernel efficiency and prefill utilization.

Prioritized by expected impact (from `docs/agent3-profile.md`, Agents 4–6):

### P0 — Make prefill actually fast (highest leverage)

**Problem:** Integrated hd=128 `_pf` cubin exists but plowrt TTFT is still
56–75% above vLLM. Segmented prefill (`plowrt_seg_final.json`) shows ~130 ms
at L≈1024 — consistent with HTTP, so the gap is real kernel time, not measurement.

**Actions:**

1. Profile one prefill launch with `ncu` / `nsys` (need admin profiling or
   `RmProfilingAdminOnly=0`) — compare against vLLM flash-attn prefill on same L
2. Tune `d_flash_prefill_mux<128,...>` occupancy and tile sizes in
   `runtime/nvidia/op_attention.cuh`
3. Enable segmented prefill cubins (`PLOW_BUILD_SEG=1` in
   `scripts/build_sm120_cubin.sh`) and wire `GpuEngine::prefill_chunk` to use
   `_pfseg` / `_pfgemm` objects for long prompts
4. Batch prefill segments (`PLOW_PF_BATCH`) when conc>1 (future; contract today is conc=1)
5. Verify vLLM and plow use **identical tokenized prompt length** (ChatML template fix)

**Target:** close TTFT gap from +56–75% to ≤0%. This is the primary win condition.

### P1 — Decode megakernel occupancy (~1.7 ms/token overhead)

**Problem:** Qwen decode step ≈6.9 ms at ctx=4096; fit intercept ≈1.7 ms unexplained.
`PLOW_NV_MINBLK=1` → ~21% peak HBM per source comments.

**Actions:**

1. Raise blocks/SM (`PLOW_NV_MINBLK=2`) if register spill stays acceptable
2. Reduce packet-gate overhead inside `interp_sm120.cu` (401 packets/token)
3. Fuse more decode ops in `devgen` (follow AddNorm / GemvQkv precedent)
4. Run `plowc tune` against `tuning/nvidia/sm_120a/` measurement DB

**Target:** recover 0.2–0.5 ms/token → closes most of the +3–7% TPOT gap.

### P2 — Flash attention utilization on 170 SMs

**Problem:** nsplit = ceil(170/32) = 6 at ctx≤8192; most SMs idle during flash decode.

**Actions:**

1. Increase `PLOW_NV_FA_GF` (grouped flash) where correctness allows
2. Revisit merge partial count vs occupancy tradeoff in `d_flash_decode<128, GF>`
3. At long ctx (32k+), nsplit grows — optimize merge kernel

**Target:** small at 4k–8k (KV is ~5% of step); matters more at 32k+.

### P3 — Runtime path (mostly done)

**Done on this branch:** `consume_prompt` — overlapped H2D, one sync after last
prompt token. Saves host time; **not** enough alone to beat vLLM.

**Remaining:**

1. CUDA graphs for prefill segment sequences (amortize launch overhead)
2. Fix ChatML template (`P5` below) before claiming any TTFT win
3. Optional: `PLOW_TTFT_LOG=1` breakdown to confirm server-side phases

### P4 — Fairness and measurement hygiene

Before declaring victory:

1. Re-run **Agent 2 contract** unchanged (`docs/agent2-benchmark-contract.md`)
2. Same client: `vllm bench serve --backend openai-chat`
3. Coherence gate: “capital of France” → contains `paris`
4. 32 prompts, 4 warmup, L ∈ {1024, 4096, 8192}, out=128, conc=1
5. Record environment in the results doc (driver, CUDA, model SHA, dtype)
6. Do **not** change benchmark code or vLLM config to slow the baseline

### P5 — Chat template (fairness, not the 27 s bug)

Qwen served through `/v1/chat/completions` currently gets Gemma turn markers unless
tokenizer probes match GLM/K3. Add a ChatML arm in `gpu_chat_prompt` so token
counts match vLLM’s Qwen template.

### P6 — Explicitly low priority for this contract

| idea | why skip (for now) |
|---|---|
| Prefix cache / VMM | contract uses `random-prefix-len=0` |
| FP8 weights/KV | vLLM fp8 slower than bf16 on 4B (`perf-data/vllm-fp8-baseline.md`) |
| CUDA graphs on decode | already 1 launch/token |
| egglog rewrite | not on GPU emit path; 0 ops reached GPU in Gemma campaign |

### Suggested work order

```
1. ChatML template fix          → fair comparison
2. ncu/nsys prefill profile     → find where +44 ms at L=1024 lives
3. Segmented prefill + kernel tune → attack P0
4. Decode occupancy (MINBLK, fusion) → attack P1
5. Agent 6 re-validation        → docs/final-validation.md, RESULT: PASS/FAIL
```

### Success criteria (Agent 6 gate)

All must hold:

1. Correctness PASS (loads, generates, oracle tests)
2. Fair benchmark PASS (identical client, GPU, model, dtype, workload)
3. **TTFT, TPOT/ITL, E2E, throughput ≤ vLLM** at L=1024, 4096, 8192
4. Reproducible (32 prompts, stable variance)

Final line in `docs/final-validation.md` must be exactly:

```
RESULT: PASS
```

---

## Quick reference commands

```bash
# Emit + serve
nix develop
plowc --hf-dir /path/to/Qwen3-4B --arch sm_120a --emit devblob+cubin --out /path/to/assets
plowrt serve --assets /path/to/assets --port 8101

# vLLM baseline (separate terminal, same GPU)
vllm serve /path/to/Qwen3-4B --dtype bfloat16 --max-model-len 16384 --port 8102

# Canonical bench client (both servers)
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8101 \
  --model qwen3-4b --tokenizer /path/to/Qwen3-4B \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --random-range-ratio 0 --request-rate inf --max-concurrency 1 \
  --num-prompts 32 --num-warmups 4 --ignore-eos --temperature 0 --seed 0
```

Replace port `8101` with `8102` for the vLLM run. Use the same flags on both sides.
