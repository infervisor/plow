# Agent 1 — Repository map

> Scope: read-only understanding of this checkout for later prefill-latency
> work. No inference, benchmark, or model behavior was changed.
>
> **Naming.** This repository is **plow** (compiler `plowc` + runtime `plowrt`),
> not a vLLM fork and not a dedicated ASR stack.

---

## 0. Scope note: “Qwen ASR text-to-text”

**FACT — there is no Qwen ASR / speech / audio encoder in this tree.**

Searches for `qwen_asr`, `QwenASR`, `whisper`, speech-to-text, and audio
frontends found none. The only “ASR” mentions are:

- `crates/plowc/src/lib.rs` — multi-network workflows described as
  `ASR → LLM → TTS`, explicitly **“Not built yet”**.
- `crates/rewrite/src/bridge.rs` — same voice-pipeline comment.

The production **text-to-text** path that later agents can actually compile,
serve, and compare to vLLM is the **Qwen3 dense GQA decoder**
(`model_type: "qwen3"` / `"qwen2_5"`). The heavily measured campaign model is
**Qwen3-4B** (`head_dim=128`, GQA 32/8, hidden 2560, 36 layers).

**FACT — Qwen3-4B geometry used throughout this map** (from comments and
emitters, not from a local checkpoint):

| field | Qwen3-4B | source |
|---|---|---|
| hidden | 2560 | `crates/plowrt/src/memory/tile_ref.rs`, `sched/mdq.rs` |
| heads / kv heads | 32 / 8 (GQA 4) | `devgen` comments; `mdq.rs` |
| head_dim | 128 (explicit, not hidden/heads) | `nn-graph` + `devgen/src/config.rs` |
| layers | 36 | `mdq.rs` |
| attn scale | `1/sqrt(128)` | `devgen/src/config.rs` `cfg_llama_qwen` |
| q/k RMSNorm | yes (`has_qk_norm`) | same |
| v_norm / k_eq_v / sliding window | no / no / none (all-global) | same |
| MLP | SwiGLU (`silu`) | same |
| tied lm_head | yes (`tie_word_embeddings`) | `runtime/tests/qwen3_sm120_chat.cu` |

**HYPOTHESIS:** the “Qwen ASR text-to-text” goal means “optimize this Qwen3
decoder prefill vs vLLM,” not add a speech encoder. If a later agent is given a
Qwen-Audio / Qwen2-Audio checkpoint, that architecture is **unsupported** today
(`nn-graph` rejects unknown `model_type`).

---

## 1. Repository architecture

### 1.1 Identity

plow compiles a Hugging Face checkpoint into a **static packet stream** and
runs it on a **persistent on-device interpreter** (one cooperative launch per
wave-class segment; counters instead of per-op CPU dispatch).

| layer | location | role |
|---|---|---|
| Compiler driver | `crates/plowc` | CLI; `--hf-dir` → `devblob` emit |
| Device-blob emitter | `crates/devgen` | **production emit** for Qwen3/Llama/Gemma |
| Symbolic model IR | `crates/nn-graph` | HF config → operator graph (analysis / tests) |
| Egglog rewrite | `crates/rewrite` | **analysis only** — not on the GPU emit path |
| Tile schedule | `crates/schedule` | packet-path scheduler (not the served Qwen blob) |
| Shared schema | `crates/plow-asset` | `weights.json` / maps compiler↔runtime |
| Device ISA | `include/packet.h`, `runtime/common/dev_isa.h`, `crates/packet` | wire + on-device instruction |
| Host runtime | `crates/plowrt` | HTTP serve, mux, CUDA/HSA engines |
| Device runtime | `runtime/nvidia`, `runtime/amd` | interpreters + kernels |
| Lean proofs | `lean-plow/` | optional compile-time checks |
| Tuning DB | `crates/tunedb`, `tuning/` | offline kernel measurements |
| Campaign numbers | `perf-data/` | plow vs vLLM writeups |

**FACT (`crates/plowc/src/lib.rs`, `crates/plowc/src/main.rs`):** egglog rewrite
is **not** in the emit path. `devgen` has no `rewrite` dependency. “Every fusion
in a shipped packet is hand-written in `devgen`.” Measured Gemma-4-12B: 0 of
1156 ops reached GPU via egglog (`perf-data/px18-egglog-wholemodel.md`).

### 1.2 Top-level tree

```
AGENTS.md, CLAUDE.md, README.md, CONTRIBUTING.md, …
Cargo.toml / Cargo.lock          workspace (15 members)
flake.nix / flake.lock           Nix: default / quantize / vllm shells
crates/                          Rust compiler + runtime
runtime/                         C / CUDA / HIP interpreters + kernels
include/packet.h                 C ABI of the wire packet
lean-plow/                       Lean 4
docs/arch/, docs/bringup/, docs/amd/, docs/flags-reference.md
perf-data/                       campaign results vs vLLM
scripts/                         emit, build, bench, vLLM compare
tools/bench/                     Cargo pin for inference-benchmarker
tuning/                          per-SKU kernel_measurement.jsonl
assets/, media/                  logos / screenshots
```

**Not present:** `shell.nix`, `pyproject.toml`, `requirements*.txt`,
`Makefile`. Python exists only inside Nix shells (`quantize`, `vllm`) and
scripts.

### 1.3 Nix environment (`flake.nix`)

Must-use environment: **`nix develop`** (devShell `default` = `plow-dev`).

| shell | purpose |
|---|---|
| `nix develop` (default) | cargo, rustc, cmake, elan; on `x86_64-linux`: ROCm 7.14 TheRock + CUDA toolkit, `PLOW_HIPCC` / `PLOW_NVCC` |
| `nix develop .#quantize` | torch + safetensors + transformers (weight quant / prep) |
| `nix develop .#vllm` | vLLM **0.27.0 client only** (`dontBuild`; `vllm bench serve` HTTP client). Marked insecure in nixpkgs; never a plow build input |

Packages: `plowc`, `plowrt` (Linux: features `cuda`+`hsa`, drivers `dlopen`’d),
`plow-runtime` (C core + ctest), `plow-interp-{sm120a,sm90a,gfx950,gfx942}`.

CI (`.github/workflows/build.yml`): self-hosted `nix develop` + `nix build` +
`cargo test --workspace`; hosted PR job is CPU `cargo check` / `fmt`.

**FACT:** `plowrt` does not link CUDA/HIP. `device::select` `dlopen`s
`libcuda.so.1` then `libhsa-runtime64`. Missing driver → CPU reference
interpreter of the **same** packet (slow, still “serves”).

**Environment limitation (this Agent 1 run):** `nix develop` and `git log`
were not executed successfully from the agent sandbox (commands rejected).
The flake contents, README, and CI file were read in full. Later agents
should enter `nix develop` before any compile/bench.

### 1.4 Qwen3 in the model zoo vs production emit

Two parallel representations:

1. **`nn-graph`** — `crates/nn-graph/src/models/qwen3.rs` `build()` /
   `build_encoder()`. Full symbolic graph: embed → N× (RMSNorm, GQA+qk-norm+RoPE,
   residual, RMSNorm, SwiGLU, residual) → final RMSNorm → lm_head (or tied
   embed). Used by rewrite tests (`fuse_qwen3`, `qwen_block_to_tiles`).
2. **`devgen` / `plowc --hf-dir`** — **this is what the GPU runs.**
   `devgen::config::cfg_llama_qwen` + `devgen::run` emit a PLOWDEV `model.pkt`.
   `plowc` `--emit` defaults to `devblob` when `--hf-dir` is set
   (`crates/plowc/src/main.rs` `run_devblob` → `devgen::run_verified`).

`plowc::hf_config::synth_llama_qwen` is the parallel metadata path for the
generic `compile()` packet pipeline. Served Qwen assets are the **devblob**.

### 1.5 Two GPU engines (not one abstracted backend)

| | NVIDIA `exec::gpu::GpuEngine` | AMD `exec::amd::AmdEngine` / `AmdServe` |
|---|---|---|
| Dispatch | one cooperative launch (optional segmented + CUDA graphs) | **n_seg launches, one drain** (AQL barrier bit) |
| Prefill vs decode objects | `interp_sm120.cubin` + `interp_sm120_pf.cubin` | `interp_decode.elf` + `interp_prefill.elf` (+ optional `interp_flash.elf`) |
| Qwen HD=128 prefill | **not in the Gemma prefill object** (see §3, §6) | **inline** via `-DPLOW_FLASH_HD128` (`scripts/build_gfx950_qwen.sh`) |
| Batching | slotted, chunk-interleaved, optional `PLOW_PF_BATCH` | compiled `PLOW_DECODE_BATCH`; idle slots still run throwaway rows |
| Prefix / VMM | CUDA VMM prefix cache | optional AMD VMM + TP prefix cache (`PLOW_PREFIX_CACHE`) |

---

## 2. Complete request execution path

Path below is **`POST /v1/chat/completions`** against `plowrt serve`. There is
**no** `/v1/completions` route (`serve/mod.rs` registers chat only).

### 2.1 Process startup

| step | location |
|---|---|
| CLI | `crates/plowrt/src/main.rs` `Cli` / `Cmd::Serve` |
| Config | `RuntimeConfig::init` (`crates/plowrt/src/config.rs`); CLI overrides env except eight env-first knobs documented in `docs/flags-reference.md` |
| Backend probe | `device::select` / `select_all` (`crates/plowrt/src/device/mod.rs`) |
| Registry | `orch::Registry::load` per `--assets` dir (`asset::ModelBundle::load`) |
| Tokenizer | `text::tokenizer::load_tokenizer` — HF `tokenizer.json` (`hf-tokenizer` feature) else `ByteTokenizer` (GPU install **refuses** byte fallback) |
| CUDA engines | `serve::manager::ModelManager::load_initial` → `GpuEngine::load` |
| AMD engines | `AmdServe::load` in `main.rs` (not S1 manager; one model, process lifetime) |
| Mux | `serve::mux::spawn` — one dispatcher task + `exec::engine_thread::EngineThread` per slug |
| HTTP | axum `serve::app` — `/v1/chat/completions`, `/v1/models`, `/healthz`, `/metrics` |

### 2.2 Model / weight / interpreter load

**CUDA (`GpuEngine::load`, `exec/gpu.rs`):**

1. Find/parse PLOWDEV blob (`asset::devblob::DevBlob`).
2. Resolve decode cubin by **ELF content** (`resolve_interp_image`, Role::Decode).
3. Load module, bind `plow_interp_sm120` (or profile symbol).
4. Optionally load prefill cubin (`load_prefill`).
5. mmap checkpoint (`asset::checkpoint::Checkpoint::open`) — every
   `*.safetensors` shard, header parsed once, zero-copy slices.
6. Tile + upload weights to HBM slab (`PLOW_WEIGHT_SLAB` default on).
7. Allocate KV rings / VMM; build per-slot tensor tables.

**AMD (`AmdEngine::load` / `load_rank`, `exec/amd.rs`):**

1. Parse blob; refuse decode/prefill object mismatch (`check_prefill_object`,
   `check_decode_object` against `.symtab`).
2. Load hsaco: `interp_prefill*.elf`, `interp_decode*.elf`, optional
   `interp_flash.elf` + `_gq` / `_fp8` twins (`object_name` / `symbol_name`).
3. Same safetensors mmap; bind by **checkpoint tensor name** (hard-fail on miss).
4. Flat `[B][kv_head][ring][hd]` KV (or VMM growth of full-attn tensors).

Tied Qwen `lm_head`: the **emitter** points the lm_head GEMV at
`model.embed_tokens.weight`. The loader never looks up `lm_head.weight`
(`qwen3_sm120_chat.cu` header).

### 2.3 Tokenizer + chat template

| step | location |
|---|---|
| Handler entry / TTFT clock | `serve/chat.rs` `chat_completions` (`t_arrive`; `obs::ttft::reset`) |
| Chat template | `gpu_chat_prompt` — probes tokenizer for `<\|end_of_msg\|>` (K3) then `<\|assistant\|>` (GLM) else **`gemma_chat_prompt`** |
| Encode | `ModelBundle::tokenizer().encode` → `HfTokenizer::encode` (`tokenizers` crate, `add_special_tokens=false`) |
| Job | `mux::Job { prompt_ids, gen, arrived, respond }` → `mux.submit` |

**FACT:** Qwen ChatML (`<|im_start|>`) is **not** a probe arm. A Qwen3 model
served through `/v1/chat/completions` gets **Gemma-4 turn markers** unless its
tokenizer encodes GLM/K3 markers as a single id (it does not).

**HYPOTHESIS:** some Qwen benches bypass this by feeding already-tokenized
ids (`qwen3_sm120_chat`, `amd-bench --prompt`) or by using vLLM’s own template
on the vLLM side only — which would make a chat-API A/B **not** like-for-like
for Qwen. Confirm before treating chat-TTFT as a Qwen correctness gate.

Image parts are **refused** (`has_image`) — text tower only.

### 2.4 Request creation → GPU

| step | location |
|---|---|
| Ingress | `mux::spawn` MPSC; capacity = engine `batch()` |
| Admission | `sched::admission::admit` (SLO / predicted wait) |
| Slot | mux slot `i` **is** engine slot `i` |
| Prefill | CUDA: `GpuEngine::begin_slot` + `prefill_chunk` / `prefill_batched`; AMD: `AmdServe::prefill` / `prefill_chunked` |
| First token | device `ARGMAX` / `ARGMAX_FIN` writes `in.ids`; host `read_sampled` / CUDA D2H of 4 bytes |
| Stream | `StreamChunk::Token` → SSE (`serve/stream.rs`, `sse_response`) or buffered JSON |
| Decode loop | `GpuEngine::step_slots_sampled` or `AmdServe::dispatch_all` |
| Stop | engine `stop_ids`, `max_tokens`, `ignore_eos` (vLLM bench extension) |

### 2.5 End-to-end call chain (AMD Qwen serve — the documented campaign path)

```
main.rs Cmd::Serve
  → Registry::load (tokenizer.json + weights.json + model.pkt)
  → AmdServe::load(blob, hsaco, checkpoint)
  → mux::spawn → dispatcher loop
chat_completions
  → gpu_chat_prompt / tokenizer.encode
  → mux.submit(Job)
dispatcher
  → AmdServe::prefill(slot, prompt_ids)          // first token
       → begin_slot
       → AmdEngine::prefill_slot
            → plan_for / plan_chunks
            → for step in chunk_steps:
                 prefill_prepare (ids/pos/kvlen HtoD, patch)
                 run_segmented     // enqueue all segs, one drain
            → read_sampled (DtoH in.ids)
  → respond first token
  → loop: AmdServe::dispatch_all
       → AmdEngine::decode_step_batched
            → decode_prepare (patch kvrow, pos/kvlen HtoD)
            → enqueue decode megakernel
            → drain
            → read_sampled
  → detok incremental delta → SSE / JSON
```

CUDA analog: `mux.rs` CUDA arm (~L1231) → `prefill_chunk` /
`gpu_prefill_batched_pass` → `step_slots_sampled` (one `cuStreamSynchronize`
per decode step).

---

## 3. Prefill execution path

### 3.1 Where prefill starts and ends

**Start (host):** first mux tick with `slot.step == 0`.

- AMD: `AmdServe::prefill` / `prefill_chunked` (`serve/engine.rs`).
- CUDA: `GpuEngine::prefill_chunk` (`exec/gpu.rs`).

**Start (device):** first interpreter launch of a **prefill-bucket program**
(`prog.t > 1`), not the T=1 decode program.

**End:** last real prompt row has been written to KV; device argmax over that
row is in `in.ids`; host returns that id as the first generated token.
Postcondition is identical to “decode-only consumption of the prompt”
(`GpuEngine::prefill_slot` docs; `qwen3_sm120_chat.cu`).

### 3.2 Bucket ladder (why prefill is not “M = prompt_len”)

AOT: row count is **part of the program identity**. Emitter builds one program
per rung. Runtime covers the prompt with a sum of rungs.

Documented in `docs/arch/13-prefill-chunking.md`.

| backend | planner | default |
|---|---|---|
| AMD | `plan_chunks` / `plan_chunks_cfg` (`exec/amd.rs`); `PLOW_RAGGED_CHUNK` default **on** → fewest-launch cover + `rebase_chunk_rows` | extra token past a rung ≈ a whole extra pass (row-invariant cost ~85% of a 128-row GLM tail — GLM number, not Qwen) |
| CUDA | `pick_prefill_bucket` (`exec/gpu.rs`); cost = `padded_rows + PLOW_PF_CHUNK_COST × launches` | `PLOW_PF_COVER=1` restores covering pick |

Qwen/Llama AMD: `scripts/build_gfx950_qwen.sh` builds **no** `interp_flash.elf`.
HD=128 flash runs **inline** in the 8-wave prefill object
(`PLOW_FLASH_HD128`). Segmented 4-wave flash is a **Gemma** occupancy split.

**FACT:** for Qwen HD=128 on gfx950, prefill is still `run_segmented` if the
packet has multiple `seg` ids, but flash does **not** switch to a 4-wave
object. Gemma needs that switch; Qwen’s build deletes `interp_flash.elf` so a
stale Gemma flash object cannot be loaded at the wrong width.

### 3.3 Per-chunk host work (AMD)

`AmdEngine::prefill_prepare` then `run_segmented`:

1. Upload chunk `ids` / `pos` / `kvlen` (pinned HtoD).
2. Patch row-count immediates (`rebase_chunk_rows` if ragged).
3. `rearm` — **zero counters once per program**, never per segment.
4. `enqueue_segment` for each `seg` (AQL packets queued **ahead**; barrier bit
   chains them; host `drain` once).
5. Repeat for next chunk.

TTFT phases: `obs/ttft.rs` (`PF_PLAN`, `PF_PREPARE`, `PF_REARM`, `PF_XCTR`,
`PF_ENQUEUE`, `PF_DRAIN`, `PF_READ`). Designed against
`vllm bench serve --max-concurrency 1`.

### 3.4 Device ops during Qwen3 prefill (dense GQA)

Emitter: `devgen` `emit_phase` for `t > 1` (not `gemv_family`). Per layer,
typical op sequence (names = `DevOp` / `PLOW_DOP_*`):

1. **RMSNORM** — input layernorm (`d_rmsnorm`, `runtime/amd/op_norm.h` /
   `runtime/nvidia/op_norm.cuh`).
2. **GEMM** family — `q_proj`, `k_proj`, `v_proj` (`d_gemm` / `d_gemm_small` /
   `d_gemm_med` / `d_gemm_wide` / `d_gemm_c5`; Qwen tile override `-DGM_BM=192`
   on gfx950 Qwen build).
3. **HEADNORM_ROPE** — per-head q/k RMSNorm + RoPE; **writes K (and V via
   separate path) into the KV cache**. Qwen: `qk_skip=0`, `v_skip=1`, real
   `v_proj`. Dispatch: `interp.hip` / `interp_sm120.cu` `PLOW_DOP_HEADNORM_ROPE`
   → `d_headnorm_rope<128>`.
4. **FLASH_PREFILL** (+ **FLASH_MERGE** on split-KV) — causal GQA flash over
   `[0, kvlen)` for the chunk’s Q rows. AMD HD=128:
   `exec_flash_prefill` → `d_flash_prefill<128>` (`op_attention.h`).
5. **GEMM** — `o_proj`.
6. residual + **RMSNORM** (prefill does **not** fuse AddNorm; `fuse_norm` is
   decode-only: `devgen` “Qwen/Llama prefill keeps the split”).
7. **GEMM_GLU** — fused gate/up SwiGLU (`d_gemm_glu`).
8. **GEMM** — `down_proj`.
9. residual.
10. After last layer: final RMSNORM + **lm_head GEMM** + **ARGMAX**.

NVIDIA prefill GEMM: `interp_sm120.cu` cases `PLOW_DOP_GEMM{,_MED,_SMALL,_GLU}`
→ `d_gemm*` in `op_gemm.cuh` / Hopper `op_gemm_sm90.cuh`.

### 3.5 Prefill vs decode — shared vs split

| | prefill | decode |
|---|---|---|
| Program | T∈ ladder (e.g. 128…8192) | T=1 (plus decode-batch rungs on AMD) |
| Interpreter object | prefill (GEMM+flash_prefill) | decode (GEMV+flash_decode) |
| Linear ops | tiled GEMM | GEMV / `GEMV_QKV` / `GEMV_GLU` |
| Attention | `FLASH_PREFILL` (+ merge) | `FLASH_DECODE` (+ `FLASH_MERGE`) |
| Norm fusion (Qwen) | split residual + RMSNorm | **AddNorm** fused (`fuse_norm`) |
| QKV fusion | separate q/k/v GEMMs | optional fused `GEMV_QKV` (bf16) |
| Occupancy (AMD Qwen) | 8-wave inline flash | 8-wave; `FA_DEC_VPIPE=8` on Qwen decode object |
| Occupancy (AMD Gemma) | 8-wave GEMM + 4-wave flash object | 8-wave |
| Host launches / chunk (AMD) | n_seg (queued ahead) | **1** |
| KV | write chunk rows; flash reads prefix+chunk | write 1 row; flash reads `kvlen` |

**FACT:** prefill and decode **share the KV layout and the HEADNORM_ROPE write
path**, not the matmul or flash kernels.

### 3.6 NVIDIA Qwen prefill — important split

`runtime/nvidia/interp_sm120.cu`:

- Default object: `PLOW_NV_FA_HD 128` — **Qwen3 decode**.
- `PLOW_NV_PREFILL` **requires** `PLOW_NV_GEMMA` (hd 256/512). FLASH_PREFILL
  dispatch traps unless `i[6]` is 256 or 512.
- Harness `runtime/tests/qwen3_sm120_chat.cu` states: **sm_120 interpreter
  cannot run Qwen prefill buckets**; prompt is consumed by the **decode
  program, one token at a time**.

`GpuEngine` **does** load a prefill cubin and run `prefill_chunk`. That path is
the **Gemma** (and other hd256/512) prefill object, not the default Qwen decode
cubin.

**FACT:** NVIDIA Qwen3-4B **correct tokens** can be produced without a prefill
kernel (decode-loop prefill). **FACT:** that is O(n) launches vs O(n/C) and is
not the AMD campaign path.

**HYPOTHESIS:** a NVIDIA Qwen prefill-object (HD=128 FLASH_PREFILL in the pf
cubin) either does not exist in the default build or is not the object
`qwen3_sm120_chat` uses. Confirm by inspecting a Qwen NVIDIA asset’s cubins
before optimizing NVIDIA Qwen prefill.

### 3.7 Sync / alloc / CPU–GPU boundary (prefill critical path)

| event | where |
|---|---|
| Host alloc | load time (weight slab, KV rings, pinned staging). Steady prefill should not allocate (`pos_stage` reused on AMD). |
| HtoD | ids/pos/kvlen per chunk; instruction patches |
| Device | entire layer stack inside interpreter(s) |
| Inter-segment | AMD: AQL barrier (GPU packet processor), **no host wait between segs**; host drain at end |
| Inter-chunk | AMD docs: currently drain per chunk (`RUNSEG`); future: queue chunks ahead |
| DtoH | 4-byte `in.ids` after last chunk |
| CUDA decode analog | patch+upload+launch+D2H then **one** `cuStreamSynchronize` (`step_slots` docs) |

---

## 4. Decode execution path

### 4.1 After first token

Mux sets `slot.step > 0`, `out_ids` holds the sampled token. Next tick:

**AMD** `AmdServe::dispatch_all`:

- Builds `pos` / `kvlen` / `parked` for all rung rows.
- Live slots: feed last token (already in `in.ids` on device for greedy —
  **do not HtoD overwrite** `in.ids`; embed reads device argmax).
- Idle slots: `pos=0, kvlen=1, id=0`, `parked=1` (throwaway compute).
- `AmdEngine::decode_step_batched` → `decode_prepare` (KV-row patch, pos/kvlen
  HtoD) → single 8-wave launch → drain → `read_sampled`.

**CUDA** `GpuEngine::step_slots_sampled`:

- Feeds `(slot, token)` for live rows; idle rows `kvlen=1` at their `pos`
  (garbage KV write lands on the row the next real step overwrites).
- Optional on-device sampler (`plow_sample`) if `temp > 0`.
- One stream sync.

### 4.2 Device ops (Qwen3 decode, measured 401 packets on sm_120 T=1)

From `interp_sm120.cu` header (Qwen3-4B decode program):

`EMBED, RMSNORM, HEADNORM_ROPE, GEMV_QKV, FLASH_DECODE, FLASH_MERGE, GEMV,
ADD_NORM, GEMV_GLU, ARGMAX, ARGMAX_FIN`

AMD decode flash: `d_flash_decode<128, PLOW_FA_GF(128)>` then
`d_flash_merge<128>` (`interp.hip`).

AddNorm: `devgen` decode-only fusion of residual+RMSNorm (Qwen/Llama).
“Deletes 72 packets/token” on that family (comment at `devgen/src/lib.rs`
~2919).

### 4.3 First token vs subsequent tokens

First token = **last prefill argmax** (or 1-token prompt: one decode step).
Subsequent tokens = decode megakernel. Same `in.ids` handshake.

---

## 5. KV cache architecture

### 5.1 Layout (device, production)

**FACT — head-major ring**, not vLLM paged token-major blocks, on the GPU
hot path:

```
K/V[kv_head][row][head_dim]
index: ((b * n_kv_head + hkv) * kv_stride + (row & kv_mask)) * D
```

Documented in `runtime/nvidia/op_attention.cuh` FLASH_DECODE contract and
`plowrt/src/memory/prefix.rs`.

| | full attention (Qwen: all layers) | sliding (Gemma only) |
|---|---|---|
| rows | `max_ctx` | `kv_ring_rows(window, chunk)` power-of-two |
| mask | `0xFFFFFFFF` (AND is no-op) | `ring-1` |
| sizing | `devgen::kv_ring` | same |

Qwen: `window=0`, all `is_full=true` → linear cache of `ctx` rows per kv-head
(`kv_ring` full branch).

Dtype: default **bf16**. Optional **e4m3 + per-row f32 scale** (`PLOW_FP8_KV`)
written by `d_headnorm_rope_fp8`, read by `d_flash_*<…, true>`.

### 5.2 Allocation

| backend | strategy | frequency |
|---|---|---|
| AMD default | one flat `[B][kvh][ring][hd]` at **load**; no paging | once |
| AMD `PLOW_VMM_KV` | VA reservation of full shape; map granules at decode frontier | grow with live ctx (full layers only) |
| CUDA VMM | growable pool + prefix cache; `ensure_rows` before a chunk | per chunk / per step frontier |
| Host `memory/kv.rs` `BlockAllocator` | vLLM-style pages for the **CPU/reference** indirection path | not the AMD Qwen HSA hot path |

AMD serve comment: **no eviction under pressure**; “eviction” = slot release
on completion. `B` is compiled `PLOW_DECODE_BATCH`.

### 5.3 Writes / reads

- **Write:** `HEADNORM_ROPE` (K, and V when not k_eq_v) at `pos`. Prefill
  writes `clen` (ragged) or full bucket rows (pad rows may write garbage past
  `real`; CUDA `ensure_rows` maps the full bucket span).
- **Read:** flash uses `in.kvlen` as the bound. `begin_slot` does **not**
  zero KV; rewinding `pos` + `kvlen` makes old rows unreachable.
- **Idle decode rows:** still write one throwaway row (AMD parked /
  CUDA kvlen=1). Correctness: that row is overwritten by the slot’s next real
  use.

### 5.4 Prefix reuse

- CUDA VMM: `vmm_attach` on first chunk; `publish` after prefill / tail.
- AMD TP: `PLOW_PREFIX_CACHE` (off by default); snapshot recurrent+KV;
  `MIN_PREFIX=128`.
- Host `PrefixCache` (`memory/prefix.rs`): RadixAttention over head-major
  runs. **FACT (same file):** FlashDecode **cannot currently read a
  multi-run (strided) prefix** — ABI limitation. Sharing that is not a single
  contiguous head-slot span is host bookkeeping, not a device gather.

### 5.5 Latency contribution

**FACT:** KV is written every prefill token and read as O(T) in prefill flash
and O(ctx) per decode token. Layout is jointly read+write optimal per
`perf-data/plow-vs-vllm-baseline.md` (campaign writeup).

**FACT:** AMD Qwen has no block table on the hot path (flat base + stride).
Indirection (`exec/indirection.rs`) is the CPU mux / design-notes path.

**HYPOTHESIS:** KV management is **not** the dominant Qwen prefill gap vs
vLLM on gfx950; campaign attributes remaining prefill gap to **flash HD=128
softmax/VALU**, not allocator churn. See §15.

---

## 6. Attention architecture

### 6.1 Implementations in-repo

| impl | files | used for |
|---|---|---|
| AMD flash prefill/decode/merge | `runtime/amd/op_attention.h`, dispatched `runtime/amd/interp.hip` | **Qwen3 AMD prefill+decode** |
| NVIDIA flash decode/merge | `runtime/nvidia/op_attention.cuh` `d_flash_decode` / `d_flash_merge` | Qwen3 NVIDIA **decode**; Gemma too |
| NVIDIA flash prefill | `d_flash_prefill` / `_mux` / `_px4` / `_px8` / `_px23`; Hopper `op_attention_sm90.cuh` | Gemma hd256/512 prefill **not** default Qwen HD=128 |
| MLA / DSA / KDA / Mamba | `op_mla.cuh`, `op_dsa.cuh`, `op_kda.h`, … | GLM / Kimi / DeepSeek — **not Qwen3** |
| CPU golden | `runtime/common` + `cpu/flash.c` | simulate / tests |
| nn-graph `nn.attention` | symbolic IR only | compile-time graph |

**No Triton kernels in plow.** Triton appears only as vLLM’s stack in
`perf-data/` (e.g. `TRITON_ATTN`).

### 6.2 Dispatch (AMD Qwen)

`interp.hip` `exec_flash_prefill`:

- `#if PLOW_FLASH_HD128`: **only** `d_flash_prefill<128>` (Qwen/Llama build).
- Else Gemma: hd 256/512, optional hd128 in 4-wave `PLOW_BUCKET_FLASH` object.

Decode: `hd == 128` → `d_flash_decode<128, PLOW_FA_GF(128)>`.

GQA: work item is `(kv_head, split)`; GF query heads share a KV row
(`devgen` GQA fusion / `nsplit` comment). Qwen GQA=4.

Causal mask: flash takes `window` immediate; Qwen `window=0` → full causal.
`scale = 1/sqrt(hd)` in the packet (`f0`), unlike Gemma’s 1.0.

RoPE: **not inside flash**. Applied in `HEADNORM_ROPE` before KV write / Q
use. Qwen: full rotary (`rope_frac_full=1.0`), `rope_theta` from config
(Qwen3 default 1e6 in `nn-graph` config; emit reads checkpoint).

### 6.3 Dispatch (NVIDIA)

Decode default: `d_flash_decode<PLOW_NV_FA_HD, PLOW_NV_FA_GF>` with
`PLOW_NV_FA_HD=128`.

Prefill: `PLOW_DOP_FLASH_PREFILL` → `d_flash_prefill_mux<256|512, …>` else
`__trap()`.

### 6.4 What the Qwen benchmark uses

**FACT (campaign docs + Qwen build script):** the Qwen3-4B vs vLLM **prefill**
numbers in `perf-data/plow-vs-vllm-baseline.md` are **AMD gfx950 / MI350X**,
flash **D=128 8-wave inline** (`qwen-prefill-perf` campaign:
“Gemma-tuned kernel ran at 2% of MFMA peak on head_dim 128 → a D=128-only
8-wave object”).

**FACT:** `FA_BKV_D128=64` was built, bit-exact, and **slower** (−30%
standalone flash; Qwen e2e 4k 145→174 ms). Shipped default `FA_BKV_D128=32`
(`op_attention.h`).

**HYPOTHESIS:** any NVIDIA Qwen prefill bench using `GpuEngine::prefill_chunk`
either runs a non-default HD=128 prefill object or falls back to decode-loop
prefill. Do not assume the Gemma `*_pf.cubin` serves Qwen3-4B.

### 6.5 Fallback paths

- Unknown `head_dim` → `__trap()` (NVIDIA) / no-op or trap (AMD depends on
  object).
- Missing flash object on Gemma → would run flash on 8-wave interpreter
  (correct, slower). Qwen build **deletes** flash.elf to avoid wrong-width load.
- CPU backend: golden kernels, not a perf path.

---

## 7. CUDA / Triton architecture

### 7.1 Custom kernels (no Triton)

NVIDIA (`runtime/nvidia/`): `interp_sm120.cu`, `interp_sm90a.cu`,
`op_gemm.cuh`, `op_gemm_sm90.cuh`, `op_attention.cuh`, `op_attention_sm90.cuh`,
`op_norm.cuh`, `op_elementwise.cuh`, `op_moe.cuh`, `op_mla.cuh`, `op_dsa.cuh`,
`op_mamba.cuh`, `sample_sm120.cu`, experiments under
`runtime/nvidia/experiments/`.

AMD (`runtime/amd/`): `interp.hip`, `op_gemm.h`, `op_attention.h`, `op_norm.h`,
`op_elementwise.h`, `op_moe.h`, `op_kda.h`, `op_k3.h`, `op_collective.h`,
`flash.hip`, `mfma.hip`.

CPU goldens: `runtime/common/interp.c` + family kernels.

Fusions (hand-written in `devgen` + kernel bodies): `GEMV_QKV`, `GEMV_GLU`,
`GEMM_GLU`, `AddNorm`, `HEADNORM_ROPE`, optional `PLOW_FUSE_HNR`,
`PLOW_FUSE_QUANT`, `PLOW_FUSE_ARGMAX` (NVIDIA; **no AMD arm** — `devgen`
audit).

### 7.2 Compilation / autotune

- NVIDIA: `scripts/build_sm120_cubin.sh`, `build_sm90a_cubin.sh`, CMake
  `nv_cubins` (`runtime/CMakeLists.txt`).
- AMD Qwen: `scripts/build_gfx950_qwen.sh` (HD=128, `GM_BM=192`,
  `FA_DEC_VPIPE=8`, global-queue objects default).
- AMD Gemma: `scripts/build_gfx950.sh` (separate 4-wave flash).
- gfx942: `scripts/build_gfx942.sh`.
- Tuner: `plowc tune`, `crates/tunedb`, `tuning/<vendor>/<isa>/<sku>/*.jsonl`.
  **Offline**; production packets do not autotune at serve time.
- `PLOW_CONFIG` generated header can drop unused opcode arms (cubin size /
  smem); `PLOW_PACKET_HASH` pairing.

### 7.3 CUDA graphs / streams / events

| mechanism | where | Qwen relevance |
|---|---|---|
| Persistent megakernel | both vendors | decode = 1 launch/token (AMD); CUDA decode similar |
| AQL barrier chaining | AMD segmented prefill | Qwen prefill segments |
| `PLOW_PF_SEG_GRAPH` | `GpuEngine` CUDA graph of segment chain (`exec/gpu.rs`) | NVIDIA segmented **Gemma** prefill; opt-in |
| `cuStreamSynchronize` | CUDA decode/prefill drain | NVIDIA serve |
| HIP/CUDA graphs in **vLLM** | `perf-data/*` | comparison baseline; plow decode “HIP graph reclaims nothing” (campaign) |

**FACT:** plow has **no** Triton JIT. **FACT:** plow decode is already
zero-launch-per-op; CUDA/HIP graphs are not the Qwen decode lever the campaign
measured.

---

## 8. Runtime architecture

### 8.1 Scheduler / batching

- Per-model mux dispatcher (`serve/mux.rs`): admit → prefill pass → decode
  launch.
- CUDA: chunk-interleave (`PLOW_PF_INTERLEAVE`); `PLOW_PF_DEFER_DECODE`
  drops decode feeds while any slot is mid-prefill; `PLOW_PF_BATCH` packs
  multi-request prefill.
- AMD: whole-prompt prefill occupies the device (unless `prefill_chunked` on
  TP); decode advances **all** compiled rows.
- Decode ladder (`PLOW_DECODE_BATCH_LADDER`): pick narrowest rung covering
  occupied slots (`sched/rungs.rs`). `batch()` stays the widest (KV sized to
  it).
- Admission: `sched/admission.rs`. Multi-step: `sched/multistep.rs`
  (`PLOW_MULTISTEP`).
- CPU reference: `sched::Scheduler::run` picks a compiled `(phase,batch,seq)`
  bucket (`sched/batching.rs`) — **not** the GPU Qwen path (GPU bundles are
  bucketless at mux; capacity from engine).

### 8.2 Threads / async

- Tokio HTTP + one mux task per model.
- `EngineThread`: dedicated OS thread so CUDA ctx / HSA queue stay bound
  (`exec/engine_thread.rs`).
- Prefetch threads at **load** (`PLOW_PREFETCH_THREADS`, default 16).
- PacketQueue (`exec/queue.rs`): designed for overlapping host bookkeeping;
  mux `queue_depth` 0 = synchronous (default comment).

### 8.3 Python overhead

**FACT:** production serve is **Rust + C/HIP/CUDA**. No Python on the Qwen
critical path. Python is benches, prep scripts, vLLM client.

### 8.4 Critical latency path (Qwen prefill, AMD, conc=1)

1. Chat template + HF BPE (host).
2. Queue / mux / engine-thread handoff.
3. `begin_slot`.
4. For each chunk: HtoD ids/pos + patch + **segmented megakernel** (GEMM +
   HD=128 flash + norms + RoPE + KV write) + drain.
5. DtoH 4-byte token + detok + SSE.

Device time dominates at 4k/8k (campaign: 148/356 ms campaign prefill vs
host microseconds). Host drain-and-refill between **Gemma** segments is a
documented ~0.5 ms × ~120 boundaries tax; **Qwen HD=128 inline flash avoids
that Gemma-specific occupancy split** but still pays per-chunk host drain.

---

## 9. Benchmark architecture

### 9.1 Do not treat every script as the Qwen harness

| entry | what it measures |
|---|---|
| `scripts/bench_plowrt_serve.sh` | `vllm bench serve --backend openai-chat` against **plowrt** OpenAI endpoint; same client as vLLM |
| `scripts/bench_vllm_chat.sh` | same client against a vLLM docker server |
| `scripts/bench_vllm_serve.sh` | thin wrapper (Gemma-4-31B defaults) |
| `scripts/twoengine/` | one Python client, both engines; TTFT/ITL/TPOT; prefix cache **off** |
| `scripts/bench_compare_vllm.sh` | crude curl throughput; default model gemma-4-12b-it |
| `scripts/plow_vs_vllm_rocm.py` | join CSVs; **Gemma/GLM pairs, not Qwen** |
| `plowrt amd-bench` | real schedule timing; tokens meaningless without `--checkpoint` |
| `runtime/tests/qwen3_sm120_chat.cu` | NVIDIA Qwen **decode-loop** correctness, not prefill perf |
| `runtime/bench/interp/qwen_interp_bench.c` | interpreter GEMM at Qwen shapes |
| `tools/bench` | empty lib; pins `inference-benchmarker` git rev |

Canonical **metric definitions** (twoengine README, aligned with
`vllm bench serve` chat backend):

| metric | definition |
|---|---|
| TTFT | time to first SSE chunk with `choices` / first content delta |
| ITL | inter-token latencies (p99 exists) |
| TPOT | `(last − first) / (out_tok − 1)` |
| throughput | completion tokens / wall |

Warmup / iterations: `BENCH_EXTRA_ARGS` (e.g. `--num-warmups 4`), `NPROMPT`,
`IN_LENS`, `CONCS`, `OUTLEN`. Defaults in `bench_plowrt_serve.sh`:
`IN_LENS=1024`, `CONCS=1`, `NPROMPT=8`, `OUTLEN=128`.

`ignore_eos`: plow accepts it so synthetic random datasets emit exactly
`max_tokens` (`GenParams`).

`PLOW_TTFT_LOG=1`: server-side breakdown (`scripts/ttft_run.sh` is GLM TP4
example).

### 9.2 Sync / timing

- Client: `vllm bench serve` (HTTP).
- Server: CUDA `cuStreamSynchronize` / HSA `drain` before token DtoH — so
  TTFT includes device completion, not just launch.
- `obs/ttft.rs` warns: global timers are **concurrency-1 only**.

---

## 10. vLLM comparison architecture

### 10.1 How comparison is supposed to work

**Same client, same protocol:** `vllm bench serve --backend openai-chat`
`--endpoint /v1/chat/completions` against plowrt **or** vLLM. Reason:
plowrt has no `/v1/completions`; mixing backends compared role-frame TTFT to
first-token TTFT (`bench_vllm_chat.sh` header).

vLLM server: docker images in scripts (e.g.
`rocm/vllm:rocm7.14.0_…_vllm_0.23.0` or older `rocm/vllm:latest` 0.11.2 in
the 2026-07-15 Qwen writeup). Nix `.#vllm` is a **0.27.0 client**, not the
ROCm server.

### 10.2 Documented Qwen3-4B vs vLLM (gfx950, bf16, batch 1)

Source: `perf-data/plow-vs-vllm-baseline.md` (refreshed 2026-07-15).

| ctx | plow main prefill ms | campaign prefill ms | vLLM docker TTFT ms | campaign/vLLM |
|---|---|---|---|---|
| 4k | 222 | **148** | 51 | **2.9×** |
| 8k | 651 | **356** | 88 | **4.0×** |

Decode campaign vs vLLM TPOT: 4k 4.7 vs 3.26 ms (**1.44×**).

**FACT:** that writeup flags a **vLLM version/method discrepancy** (0.25.1
in-process vs 0.11.2 docker served) and says a controlled same-version A/B
is still the fix.

**FACT:** campaign branches named `qwen-prefill-perf` / `qwen-decode-perf`
are described as **not yet merged to main** at the time of that note.

Correctness bar in that doc: full-completion HF token match.

### 10.3 Structural difference vs vLLM (prefill)

From `docs/arch/13-prefill-chunking.md`: vLLM token budget is a **maximum**
(exact M at runtime). plow M is a **compiled quantum**. Neither is universally
better; plow fails at rung tails, vLLM pays per-op launch.

vLLM: Triton/FA3/AITER attention, CUDA/HIP graphs, paged KV.
plow: persistent interpreter, head-major KV, AOT tiles.

---

## 11. Important configuration variables

Emit / build (Qwen AMD prefill):

| knob | effect |
|---|---|
| `plowc --hf-dir --arch gfx950 --gpu mi350x\|mi355x --max-ctx … --out` | production emit |
| `PLOW_FLASH_HD128` | compile HD=128 flash into 8-wave prefill object |
| `GM_BM=192` | Qwen prefill GEMM tile (build script) |
| `FA_DEC_VPIPE=8` | Qwen decode V-prefetch |
| `PLOW_NO_GQ=1` | static scheduler objects instead of global queue |
| `PLOW_FP8` / `PLOW_FP8_KV` | fp8 weights / fp8 KV objects |
| `PLOW_RAGGED_CHUNK` | default on; `=0` padding DP |
| `PLOW_UNISEG=1` | **NVIDIA-only**; on AMD collapses wave-class segs and **breaks prefill** (README) |
| `PLOW_L2_PLACE` | default on gfx942; gfx950 opt-in; skipped on multi-class prefill programs |

Runtime:

| knob | effect |
|---|---|
| `PLOW_CHECKPOINT` / `--rt-checkpoint` | weight dir |
| `PLOW_HSACO` | AMD code objects |
| `PLOW_TTFT_LOG` | TTFT breakdown |
| `PLOW_PF_INTERLEAVE` | CUDA chunk interleave cap |
| `PLOW_PF_BATCH` | CUDA batched prefill |
| `PLOW_PF_CHUNK_COST` / `PLOW_PF_COVER` | CUDA bucket pick |
| `PLOW_DECODE_BATCH` / `PLOW_DECODE_BATCH_LADDER` | emit-time B / rungs |
| `PLOW_PREFIX_CACHE` | AMD TP prefix |
| `PLOW_VMM_KV` | AMD growable KV |
| `PLOW_WEIGHT_SLAB` | single weight allocation (default on) |

Full list: `docs/flags-reference.md`, `crates/plowrt/src/config.rs`,
`devgen` `EmitConfig`.

---

## 12. Important source files

| area | files |
|---|---|
| Qwen graph | `crates/nn-graph/src/models/qwen3.rs`, `config/qwen3.rs` |
| Qwen emit | `crates/devgen/src/config.rs`, `crates/devgen/src/lib.rs` (`run`, `emit_phase`, `kv_ring`) |
| Compiler CLI | `crates/plowc/src/main.rs` (`run_devblob`), `hf_config.rs` |
| Serve | `plowrt/src/main.rs`, `serve/chat.rs`, `serve/mux.rs`, `serve/engine.rs` |
| AMD engine | `plowrt/src/exec/amd.rs`, `amd_tp.rs` |
| CUDA engine | `plowrt/src/exec/gpu.rs` |
| Tokenizer | `plowrt/src/text/tokenizer.rs` |
| KV host | `plowrt/src/memory/kv.rs`, `prefix.rs`, `vmm.rs` |
| Weights | `plowrt/src/asset/checkpoint.rs`, `memory/weights.rs` |
| AMD interp / attn / gemm / norm | `runtime/amd/interp.hip`, `op_attention.h`, `op_gemm.h`, `op_norm.h` |
| NVIDIA interp / attn | `runtime/nvidia/interp_sm120.cu`, `op_attention.cuh`, `op_norm.cuh`, `op_gemm.cuh` |
| ISA | `runtime/common/dev_isa.h`, `include/packet.h` |
| Qwen NVIDIA harness | `runtime/tests/qwen3_sm120_chat.cu` |
| Qwen AMD build | `scripts/build_gfx950_qwen.sh` |
| vLLM compare | `scripts/bench_plowrt_serve.sh`, `bench_vllm_chat.sh`, `scripts/twoengine/` |
| Qwen vs vLLM numbers | `perf-data/plow-vs-vllm-baseline.md` |
| Prefill design | `docs/arch/13-prefill-chunking.md`, `docs/arch/06-runtime.md` |

---

## 13. Important functions / classes

| symbol | role |
|---|---|
| `nn_graph::models::qwen3::build` | symbolic Qwen3 decoder graph |
| `devgen::config::cfg_llama_qwen` | checkpoint → emit `Cfg` (qk-norm, SwiGLU, 1/sqrt(hd)) |
| `devgen::run` / `run_verified` | emit `model.pkt` |
| `devgen::kv_ring` | KV rows + mask |
| `plowc::main::run_devblob` | `--hf-dir` production compile |
| `plowrt::serve::chat::chat_completions` | HTTP entry |
| `gpu_chat_prompt` / `gemma_chat_prompt` | template (Qwen falls through to Gemma) |
| `mux::spawn` / dispatcher loop | continuous batching |
| `AmdServe::load` / `prefill` / `dispatch_all` | AMD serve |
| `AmdEngine::load` / `prefill` / `prefill_prepare` / `run_segmented` / `decode_step` | AMD device driver |
| `GpuEngine::load` / `prefill_chunk` / `pick_prefill_bucket` / `step_slots_sampled` | CUDA device driver |
| `Checkpoint::open` | safetensors mmap |
| `d_flash_prefill<128>` / `d_flash_decode<128>` | AMD Qwen attention |
| `d_headnorm_rope<128>` | RMSNorm + RoPE + KV write |
| `d_gemm*` / `d_gemm_glu` | prefill matmuls |
| `d_gemv*` / `GEMV_QKV` / `GEMV_GLU` | decode matmuls |
| `d_rmsnorm` / `AddNorm` | norms |
| `obs::ttft::*` | TTFT phases |

---

## 14. Existing optimization work

**Git history:** `git log` was not available in this agent environment.
The following is reconstructed from **committed comments, scripts, and
`perf-data/`**. Later agents should run `git log --all --grep=qwen` and
inspect branches `qwen-prefill-perf` / `qwen-decode-perf` if they still exist.

### 14.1 Qwen-specific (documented)

| attempt | result (as recorded in-tree) |
|---|---|
| HD=128 flash object (8-wave inline, `PLOW_FLASH_HD128`) | campaign: large prefill win vs Gemma-tuned D=256/512 kernel at 2% MFMA peak |
| `GM_BM=192` vs 256 | ~1% e2e prefill; register-legal; shipped in `build_gfx950_qwen.sh` |
| `GM_SLICE=32` | +7–9% standalone GEMM, **neutral** e2e; left at 16 |
| `FA_BKV_D128=64` | exact, fits, **−30%** flash; not shipped |
| `FA_DEC_VPIPE=8` | Qwen decode V-prefetch; Gemma build leaves 0 |
| AddNorm (Qwen/Llama decode) | packet/gate reduction; decode campaign modest close |
| Exact-tiling GEMV | listed in campaign “verified wins” |
| flash_decode nsplit | ~5% decode; “real decode win” |
| QKV fusion bf16 | bs=1-neutral / sometimes slower (coarsens CU map) |
| `PLOW_FUSE_QKV_FP8` | measured slower on Gemma fp8; off by default |
| Global queue (`_gq` objects) | default; decode win, prefill neutral (build script) |
| HIP graphs on plow decode | “reclaims nothing” (already 1 dispatch/token) |
| fp8 for Qwen3-4B vs vLLM | `perf-data/vllm-fp8-baseline.md`: vLLM fp8 **slower** than vLLM bf16 on 4B; plow-fp8 should compete with vLLM **bf16** |

### 14.2 Prefill / attention / graphs (broader, still in tree)

- Ragged-M chunking (`PLOW_RAGGED_CHUNK`).
- CUDA `pick_prefill_bucket` launch-cost DP (8190 vs 8390 row anecdote).
- NVIDIA FA pipeline (`PLOW_NV_FA_PIPE`, px4/px8/px23) — Gemma hd256/512.
- CUDA segment graphs (`PLOW_PF_SEG_GRAPH`).
- Counter double-buffer (`ctr_dbuf`) to hide re-arm behind decode.
- `runtime/nvidia/experiments/` — dozens of A/B probes (TMA, wgmma, splitzip,
  warpspec, VMM KV, …), mostly NVIDIA Gemma/H100/sm_120.
- `runtime/ubench/*` — Qwen3-4B prefill GEMM occupancy/MFMA shape verdicts.

### 14.3 TODOs / incomplete (Qwen-relevant)

- ASR→LLM→TTS workflow: not built.
- NVIDIA Qwen **prefill kernels** (hd128 FLASH_PREFILL): not in default pf
  object; harness uses decode loop.
- Chat template: no Qwen/ChatML arm.
- Egglog fusions never reach GPU.
- `PLOW_FUSE_ARGMAX` has no AMD arm.

---

## 15. Potential bottlenecks

Each item is tagged **FACT** (measured or mechanically true in this tree) or
**HYPOTHESIS** (plausible, not verified here).

### Prefill (primary target)

1. **FACT:** On gfx950 Qwen3-4B, campaign prefill is still **2.9–4.0×** vLLM
   docker TTFT at 4k/8k after HD=128 flash + GEMM tile work
   (`plow-vs-vllm-baseline.md`).
2. **FACT (campaign attribution):** remaining prefill is **~78% attention
   (flash), softmax-VALU-bound**; MFMA **1.8%** of issue at D=128; fp8 “barely
   helps” that term.
3. **FACT:** D=128 `FA_BKV=64` was tried and lost. Do not re-enable without
   new evidence.
4. **FACT:** AOT bucket ladder cannot express arbitrary M; a 1-token tail
   can cost most of a pass (`docs/arch/13-prefill-chunking.md`). Ragged-M is
   already default on AMD.
5. **FACT:** Qwen prefill still uses **split** residual+norm (AddNorm is
   decode-only).
6. **HYPOTHESIS:** further prefill wins are algorithmic in HD=128 flash
   (softmax/exp/barrier), not more GEMM tiling. Campaign leftover GEMM lever
   cited: 192×256 already shipped; “+15% power-aware” 192×256 in 8-GPU regime
   is a **campaign claim**, not re-measured here.
7. **HYPOTHESIS:** host per-chunk drain is second-order vs flash at 4k+;
   still visible at short prompts. Not re-timed in this pass.
8. **FACT:** NVIDIA Qwen prefill-as-GEMM/FLASH_PREFILL is not the default
   sm_120 object. Optimizing NVIDIA Qwen TTFT via `GpuEngine::prefill_chunk`
   without a HD=128 pf cubin will not hit GEMM/flash_prefill.

### Decode (secondary)

9. **FACT:** campaign decode still **~1.44×** vLLM TPOT at 4k.
10. **FACT (campaign):** decode is **overhead/occupancy-bound, not HBM-bound**
    (Qwen 27% of HBM roofline vs vLLM 42%). vLLM edge = fewer/larger fused ops,
    not graphs.
11. **FACT:** idle slots in a wide `PLOW_DECODE_BATCH` still execute throwaway
    rows. Ladder exists to bound that waste.
12. **HYPOTHESIS:** AddNorm + nsplit already harvested the easy decode packet
    reductions for Qwen.

### Runtime / correctness traps (can look like “perf”)

13. **FACT:** missing HSA `dlopen` → CPU interpreter, fluent answers, fictional
    timings (`scripts/twoengine/README.md` gate 2).
14. **FACT:** Qwen chat template is Gemma’s on `/v1/chat/completions`.
15. **FACT:** `PLOW_UNISEG=1` on AMD breaks prefill.
16. **FACT:** byte-fallback tokenizer is refused on GPU install; a CPU fallback
    serve is silent garbage if that check is skipped.
17. **HYPOTHESIS:** prefix-cache-on vLLM vs prefix-cache-off plow would be
    mis-attributed as a kernel gap (twoengine docs; they disable both).

### KV

18. **FACT:** Qwen full-attn KV is a linear `ctx` ring per kv-head, allocated
    at load (unless VMM). No paged gather on the flash hot path.
19. **HYPOTHESIS:** KV capacity/ fragmentation is a **concurrency / max_ctx**
    issue, not the conc=1 TTFT gap.

---

## 16. Verification of this document

- Path: `docs/agent1-repository-map.md` (this file).
- Content: non-empty architecture + request/prefill/decode/KV/attention/CUDA/
  runtime/benchmark/vLLM/config/source/function/optimization/bottleneck map.
- Inference, benchmarks, and model code were **not** modified.

**Follow-up for Agent 2+:** enter `nix develop`; confirm Qwen3-4B assets +
`build_gfx950_qwen.sh` objects; treat AMD HD=128 `d_flash_prefill` as the
prefill optimization surface unless a new NVIDIA HD=128 pf object is in
scope; do not assume a Qwen ASR encoder exists.
