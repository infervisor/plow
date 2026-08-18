# Agent 3 — Qwen3-4B NVIDIA (sm_120) latency profile

> Scope: identify where TTFT / prefill / ITL / decode / E2E latency is spent.
> **No production code was changed. No kernels were optimized. Benchmark
> semantics were not changed.**
>
> Contract: `docs/agent2-benchmark-contract.md`. Repository map:
> `docs/agent1-repository-map.md`. Prior A/B: `docs/baseline-results.md`
> (HTTP cells still empty).
>
> Every numeric claim is tagged **MEASURED FACT**, **SOURCE FACT**,
> **DERIVED**, or **HYPOTHESIS**. “MEASURED FACT (prior, this GPU)” means
> numbers committed in this tree from RTX 5090 / Qwen3-4B decode steps —
> not a new HTTP run in this Agent 3 session.

---

## 0. What this session could and could not measure

### 0.1 This-session hardware probes

| probe | result |
|---|---|
| `/dev/nvidia2`, sysfs GPU name | **MEASURED FACT:** 1× GeForce RTX 5090, PCI `0000:41:00.0`, minor 2 |
| `/proc/driver/nvidia/version` | **MEASURED FACT:** driver **580.142** Open Kernel Module |
| `/usr/local/cuda/include/cuda.h` | **MEASURED FACT:** `CUDA_VERSION 13000` (CUDA 13.0) |
| conda `torch/version.py` | **MEASURED FACT:** PyTorch **2.9.0+cu130** (host; **not** on plow serve path) |
| `/proc/driver/nvidia/params` | **MEASURED FACT:** `RmProfilingAdminOnly=1` |
| `/usr/local/cuda/bin` | **MEASURED FACT:** `ncu` present; **`nsys` absent** |
| Qwen3-4B weights | **MEASURED FACT:** none under `/root/models`, `/workspace/models`, HF cache |
| `target/release/plowrt` | **MEASURED FACT:** no `target/` in this worktree |
| `nix develop`, `nvidia-smi`, `python3 -c`, `git status` | **MEASURED FACT:** Auto-review **rejected** in this sandbox |
| Canonical `vllm bench serve` cells | **NOT RUN** (same blockers as Agent 2, plus no binary/weights) |

Do not treat gfx950 campaign TTFT as this baseline (`docs/baseline-results.md` Appendix A).

### 0.2 Strongest existing measurements on **this** GPU

Committed in `crates/plowrt/src/bin/qsim.rs` and `crates/plowrt/src/sched/mdq.rs`:

**MEASURED FACT (prior, this GPU):** Qwen3-4B decode, RTX 5090 sm_120a, batch=1, 143 timed steps:

| compiled ctx | mean step ms | source |
|---:|---:|---|
| 4096 | **6.8699** | sd 0.0105, min 6.845, max 6.895, CV **0.15%** |
| 8192 | **7.1264** | qsim measured table |
| 16384 | **7.9167** | qsim measured table |
| 32768 | **9.5588** | qsim measured table |

Fit (`ServiceTable::fit`): effective HBM **1673 GB/s**, fixed overhead **1.70 ms/step**.
Checkpoint bytes in that model: **8_045_000_000** (~7.49 GiB). KV bytes/token:
`2 × 8 × 128 × 2 × 36 = 147_456`. Packets/token: **401**.

`interp_sm120.cu` header independently records the same 401-packet T=1 decode program.

---

## 1. Profiling methodology

Intended (Agent 2, conc=1):

1. `nix develop` → CUDA `plowrt serve` (private binary, `libcuda` present).
2. `PLOW_TTFT_LOG=1` on one conc=1 pass (server phases).
3. `PLOW_STEP_TIME=1` CUDA-event split (upload / interp / D2H) inside `GpuEngine::step_slots_sampled`.
4. Canonical client: `nix develop .#vllm --command vllm bench serve --backend openai-chat …` at L∈{1024,4096,8192}.
5. Device: `ncu` / `nsys` around one representative request.

What actually happened:

- Steps 1–5 **did not run** (no weights, no plowrt, Auto-review, no nsys, ncu counters admin-only).
- Critical path was reconstructed from **SOURCE FACT** (serve → mux → `step_slots_sampled` → `interp_sm120`) plus the **prior RTX 5090 decode-step table**.
- Prefill/TTFT/E2E numbers below for NVIDIA Qwen are **DERIVED** from that table under the **SOURCE FACT** that sm_120 Qwen prompt consumption is the T=1 decode program, one launch per prompt token.

Diagnostic that still exists and does **not** change bench semantics (for a later exclusive box):

| tool | what it times | notes |
|---|---|---|
| `PLOW_TTFT_LOG=1` | handler → first SSE | conc=1 only (`obs/ttft.rs`) |
| `PLOW_STEP_TIME=1` | CUDA events per decode step | `GpuEngine` `StepTiming` |
| `step_bench` example | engine-only, no HTTP | **not** the contract client |
| `PLOW_NV_SKELETON=1` cubin | gate/signal with no op bodies | measurement-only; garbage tokens |
| `ncu --set full` | SM counters | **blocked** here by `RmProfilingAdminOnly=1` |

---

## 2. Hardware

| item | value | tag |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB, PCI `0000:41:00.0` (`/dev/nvidia2`) | MEASURED FACT |
| Other host 5090s | `c1:00.0`, `81:00.0` **not** passed through | MEASURED FACT (sysfs) |
| Driver | 580.142 | MEASURED FACT |
| Host CUDA | 13.0 (`CUDA_VERSION 13000`), `nvcc` at `/usr/local/cuda/bin/nvcc` | MEASURED FACT |
| Host PyTorch | 2.9.0+cu130 — **not** plow serve | MEASURED FACT |
| Power | `power_state=D0` | MEASURED FACT |
| Clocks | **not read** (`nvidia-smi` rejected) | — |
| MIG | off (`/dev/nvidia-caps` empty) | MEASURED FACT |
| AMD | none (`/dev/kfd` absent; Agent 2) | SOURCE FACT |
| ncu admin | `RmProfilingAdminOnly=1` | MEASURED FACT |
| SM count used by packets | comments/harness: **170** SMs on this 5090 | SOURCE FACT (`qwen3_sm120_chat.cu`) |

Nix flake (unread at runtime this session): `nixpkgs` pin in Agent 2 contract `49a4bd0573c376468dd7996ddb6f9fa31d8c4d97`; CUDA via `PLOW_NVCC` inside `nix develop`.

Relevant env **contract** (leave unset unless documented): `PLOW_VMM_PREFIX` off, `PLOW_PREFIX_CACHE` off, `PLOW_PF_BATCH` default, `CUDA_VISIBLE_DEVICES` same single index. **This process:** Agent 2 recorded them unset; this session could not dump `/proc/self/environ`.

---

## 3. Benchmark configuration

Canonical argv: Agent 2 §5.1. **Not executed.**

| knob | contract |
|---|---|
| Model | Qwen/Qwen3-4B bf16, no quant |
| Client | `vllm bench serve --backend openai-chat` |
| L | 1024, 4096, 8192 |
| out | 128, `--ignore-eos`, temp 0, seed 0 |
| n / warmup / conc | 32 / 4 / 1 |
| plow prefill path | **decode-loop** unless a verified hd=128 `_pf` cubin exists |

**SOURCE FACT:** default `interp_sm120_pf.cubin` implements `FLASH_PREFILL` only for hd **256/512**; hd 128 **`__trap()`s** (`interp_sm120.cu` `PLOW_DOP_FLASH_PREFILL`). If `_pf` is **absent**, mux uses `gpu_prefill_advance` else-branch: one T=1 launch per prompt token. If `_pf` is **present**, serve **must not** be timed (trap). That **is** the current NVIDIA Qwen implementation.

---

## 4. CPU profile

**SOURCE FACT:** production serve is Rust + CUDA driver. **No Python** on the Qwen critical path.

### 4.1 TTFT host chain (request accepted → first SSE)

```
axum accept + JSON          UNACCOUNTED vs PLOW_TTFT_LOG (client includes it)
chat_completions t_arrive
  gpu_chat_prompt           TEMPLATE   — Qwen falls through to Gemma-4 markers
  tokenizer.encode          ENCODE     — HF BPE, add_special_tokens=false
  mux.submit                QUEUE
  EngineThread:
    begin_slot
    for each prompt token:  PREFILL (NVIDIA Qwen = decode loop)
      patch kvrow (B=1, unless plow_dyn_kvrow)
      H2D ids/pos/kvlen     pinned, no alloc
      memset counters
      cooperative launch
      D2H in.ids (4 B)
      cuStreamSynchronize   ← token not visible to host before this
    detok + channel send    FIRST_TOK
  sse_response              role delta rides first token chunk
```

Locations: `serve/chat.rs`, `obs/ttft.rs`, `serve/mux.rs` `gpu_prefill_advance`, `exec/gpu.rs` `step_slots_sampled`.

**NOT MEASURED THIS SESSION:** phase milliseconds. `PLOW_TTFT_LOG` CUDA arm does **not** currently fill AMD-only `PF_ENQUEUE`/`PF_DRAIN` rows; NVIDIA time lands in `PREFILL` as the mux-visible `gpu_prefill_advance` wall.

### 4.2 Per-token host work (decode)

**SOURCE FACT** (`step_slots_sampled`): no per-step heap alloc. Pinned staging reused. Submission is async; **one** `cuStreamSynchronize` retires patch+H2D+memset+launch+D2H.

**HYPOTHESIS:** host enqueue is microseconds; `sync_wait` ≈ device interp time. Supported by `StepTiming` design (gap / submit / sync + CUDA events) but **not re-logged here**.

**SOURCE FACT:** greedy `temp<=0` does **not** launch `plow_sample`. Contract client sends `temperature: 0`.

---

## 5. GPU profile

### 5.1 Launch model

**SOURCE FACT:** Qwen3-4B decode is **one cooperative kernel** per token, not one CUDA launch per op.

| item | value |
|---|---|
| Kernel | `interp_sm120` (mangled `_Z12interp_sm12011PlowProgram`) |
| File | `runtime/nvidia/interp_sm120.cu` |
| Grid | `n_cu` (170 on this 5090) × block 256 |
| Occupancy target | `PLOW_NV_MINBLK=1` → **1 block/SM** |
| Scheduler | `PLOW_NV_SCHED=1` global queue (default) |
| Packets / launch | **401** (MEASURED FACT, program dump) |
| Sync | one stream sync per `step_slots` |

**SOURCE FACT:** CUDA graphs are not on this Qwen decode path. AMD campaign: “HIP graphs reclaim nothing” for the same 1-dispatch/token reason.

### 5.2 Bound (decode, conc=1) — from the fitted byte model

Using **MEASURED** step times + **SOURCE** byte counts (not ncu):

At ctx **4096**, mean **6.8699 ms**:

| term | ms | % of step | tag |
|---|---:|---:|---|
| Weight stream 8.045 GB @ 1673 GB/s | **4.81** | **70%** | DERIVED from fit |
| KV stream 147456×4096 B | **0.36** | **5%** | DERIVED |
| Fixed (gates / occ / launch) | **1.70** | **25%** | MEASURED FACT (fit intercept) |

**MEASURED FACT (source comment, `interp_sm120.cu`):** decode object is “bandwidth-STARVED at 1 block/SM (12.5% occ, ~21% of peak HBM)” in isolated GEMV occupancy notes. That **does not contradict** the 1673 GB/s fit: the fit **folds occupancy/gates into `fixed_ms`**, then attributes the rest to bytes.

Verdict for **decode ITL** (evidence = affine fit + comments, **not** ncu):

| bound | decode? | evidence |
|---|---|---|
| Memory (weights) | **primary** | 70% of 6.87 ms is 7.49 GiB / fitted BW |
| Occupancy / packet-gate | **secondary, large** | 1.70 ms intercept; 401 counters; 1 blk/SM |
| Compute (tensor-core GEMM) | **no** | decode uses **GEMV**, not tiled GEMM; comment “MFMA-free” |
| CUDA launch-bound (many kernels) | **no** | 1 launch/token |
| Host-sync bound beyond kernel | **no** (HYPOTHESIS) | sync waits on the interp; CV 0.15% |
| CPU-bound | **no** at 6.9 ms/token | host submit is designed as enqueue-only |

Verdict for **NVIDIA Qwen “prefill”** (decode-loop):

| bound | ? | evidence |
|---|---|---|
| Repeat weight-stream × L | **yes, dominant** | each prompt token re-reads 7.49 GiB |
| Launch-count | **yes, structural** | L cooperative launches vs O(L/C) GEMM prefill |
| Flash-prefill compute | **N/A** | hd128 FLASH_PREFILL not in default object |

---

## 6. Prefill profile (highest priority)

**SOURCE FACT:** there is **no** Qwen hd=128 `FLASH_PREFILL` / `GEMM_*` path on default sm_120. Prefill **is** decode.

### 6.1 Kernel count (NVIDIA Qwen, one request, prompt L, out 128)

| window | CUDA launches of `interp_sm120` | tag |
|---|---:|---|
| Prompt consumption | **L** | SOURCE FACT (`gpu_prefill_advance` loop) |
| First token | included in last prompt step (device ARGMAX_FIN) | SOURCE FACT |
| Decode tokens 2..128 | **127** | SOURCE FACT |
| Total | **L + 127** | DERIVED |

Internal device ops per launch: **401 packets**. That is **not** 401 `cuLaunchKernel`s.

### 6.2 GPU time (DERIVED from measured decode affine model)

`step_ms(ctx) = 1.70 + 1000×(8.045e9 + 147456×ctx)/1e9/1673`

Prompt walk, kvlen = 1..L:

`T_prefill(L) ≈ L×6.5087 + 0.00008814×L×(L+1)/2` ms

| L | DERIVED device+sync prefill | vs vLLM docker TTFT (gfx950, **not this contract**) |
|---:|---:|---|
| 1024 | **~6.71 s** | 20.63 ms (different GPU/instrument — do not ratio) |
| 4096 | **~27.4 s** | 50.51 ms |
| 8192 | **~56.3 s** | 87.63 ms |

**HYPOTHESIS (high confidence):** RTX 5090 HTTP TTFT at L=4096 will be **tens of seconds**, not tens of milliseconds, until an hd=128 prefill object exists. This is the implementation, not a client bug (Agent 2 §8.4).

### 6.3 Largest “kernels” inside one launch

ncu **not run**. Attribution from **SOURCE FACT** (opcode mix) + byte model:

| device op | role | likely share of 6.9 ms | tag |
|---|---|---|---|
| `GEMV_QKV` + `GEMV` o/down + `GEMV_GLU` | stream almost all weights | **majority (~weight term)** | HYPOTHESIS, high |
| `FLASH_DECODE` + `FLASH_MERGE` | KV read O(ctx) | **grows with ctx** (0.36 ms @4k in byte model) | DERIVED |
| `HEADNORM_ROPE` | q/k RMSNorm + RoPE + **KV write** | small vs GEMV | HYPOTHESIS |
| `RMSNORM` / `ADD_NORM` | 1-row reductions; other SMs wait | part of `fixed_ms` | HYPOTHESIS |
| `EMBED`, `ARGMAX`, `ARGMAX_FIN` | tiny vs 7.5 GiB | negligible | HYPOTHESIS |
| Counter gate/signal × 401 | interpreter tax | **inside 1.70 ms** | HYPOTHESIS, high |

**Idle GPU:** **SOURCE FACT** — a decode RMSNorm is one row on one block while other blocks poll counters (`devgen` emit_phase comments). Flash nsplit at n_cu=170, 32 heads, ctx≤8192: `ns = ceil(170/32) = 6` → far fewer than 170 flash work-items. **HYPOTHESIS:** flash underfills the 5090; GEMV is issued on `all` CUs.

**Synchronization:** one host `cuStreamSynchronize` **per prompt token**. Device-side: per-packet acquire/release (`PLOW_NV_PTXSYNC=1`).

**Memory alloc during prefill:** **SOURCE FACT** — load-time slab + pinned `StepStage`. Steady `step_slots` does not `cudaMalloc`.

---

## 7. TTFT profile

**SOURCE FACT (metric):** client TTFT = HTTP POST → first SSE `choices` chunk. plow `sse_response` puts `role: assistant` on that **same** first-token chunk (do not reintroduce a leading role frame).

### 7.1 Critical path (NVIDIA Qwen)

1. HTTP + JSON (client-visible, not in `PLOW_TTFT_LOG`).
2. Gemma-4 chat template + HF tokenize (**unnecessary vs ChatML**, but **small** vs seconds of decode-loop).
3. Mux / engine-thread handoff.
4. **`for t in prompt_ids: step_slots`** — **the** TTFT cost.
5. D2H last `in.ids` + detok + SSE.

### 7.2 Unnecessary work on the TTFT path

| work | why it is on the path | cost | tag |
|---|---|---|---|
| Re-read **7.49 GiB weights L times** | no hd128 GEMM prefill | **~4.81 ms × L** | DERIVED |
| 401 packet gates × L | megakernel decode program | **~1.70 ms × L** | DERIVED |
| Host sync + D2H after **every** prompt token | `step_slots` always D2H; only **last** token is used | D2H 4 B is tiny; sync is the kernel | SOURCE FACT + HYPOTHESIS |
| ChatML missing | Gemma wrap | milliseconds | SOURCE FACT (Agent 1/2) |
| `plow_sample` | off at temp=0 | 0 | SOURCE FACT |

**HYPOTHESIS:** collapsing the prompt loop into `PLOW_MULTISTEP`-style enqueue (one sync at end) saves host gaps, **not** the 4.81 ms×L weight term. Second-order vs a real prefill object.

---

## 8. ITL / decode profile

**MEASURED FACT (prior, this GPU), after KV already at ctx:**

| ctx | ms/token | tok/s | tag |
|---:|---:|---:|---|
| 4096 | 6.8699 | **145.6** | MEASURED |
| 8192 | 7.1264 | **140.3** | MEASURED |
| 16384 | 7.9167 | **126.3** | MEASURED |

Contract ITL is the gap **token 1 → token 2 …** after TTFT, conc=1, 127 gaps/request. At L=4096, ITL ≈ 6.87 ms (ctx grows +127; **DERIVED** extra ≈ 0.01 ms — negligible).

Kernel sequence **per ITL sample** (SOURCE FACT, T=1 program):

`EMBED → RMSNORM (layer0) → [GEMV_QKV → HEADNORM_ROPE → FLASH_DECODE → FLASH_MERGE → GEMV o → ADD_NORM → GEMV_GLU → GEMV down → ADD_NORM] × 36 → final RMS + lm_head GEMV → ARGMAX → ARGMAX_FIN`

(`fuse_norm` folds residual+RMSNorm on Qwen decode; `fuse_qkv` is on for bf16.)

KV: `HEADNORM_ROPE` writes 1 row; `FLASH_DECODE` reads `kvlen` rows, head-major `[kv_head][row][128]` bf16.

Memory movement/step @ ctx 4096: **~8.65 GB** (weights + KV). Launch overhead: 1 cooperative launch. Sync: 1 stream sync.

---

## 9. Memory profile

| pool | size | when | tag |
|---|---|---|---|
| Weights | **7.49 GiB** bf16 (`8_045_000_000` B) | load, `PLOW_WEIGHT_SLAB` default on | MEASURED FACT (mdq) + SOURCE |
| KV | `147_456 × max_ctx` B | load if no VMM; else grow | SOURCE FACT |
| KV @ max_ctx 16384 | **~2.25 GiB** | contract compile | DERIVED |
| KV @ live 4096 / 8192 | 0.56 / 1.13 GiB | — | DERIVED |
| Activations (T=1) | hidden/vocab rows, not 7 GiB | load | SOURCE FACT |
| Pinned staging | `3 × batch × 4` B | load, reused | SOURCE FACT |
| Counters | `n_counter` u32, memset every step | load; **zeroed per launch** | SOURCE FACT |
| Temp allocs on hot path | **none** | `step_slots` | SOURCE FACT |
| Fragmentation | not observed | no ncu/cudaMemGetInfo | NOT MEASURED |

**SOURCE FACT:** `PLOW_VMM_PREFIX` default **off** (contract). CUDA VMM KV only if that bring-up runs.

Peak VRAM conc=1, max_ctx 16k: **HYPOTHESIS ≈ 10 GiB** (weights + full KV + small workspace) of 32 GiB — **not** the TTFT limiter.

Allocation frequency: **once at load**, plus per-step memset of counters (not a new allocation).

---

## 10. Hottest kernels → source

ncu launch counts/times **NOT MEASURED THIS SESSION**. “Launches” below are CUDA launches of the **megakernel**, not per-op.

### 10.1 The only CUDA kernel on the greedy Qwen path

| field | value |
|---|---|
| Kernel name | `interp_sm120` / `_Z12interp_sm12011PlowProgram` |
| Source | `runtime/nvidia/interp_sm120.cu` (`PLOW_SYM(interp_sm120)`) |
| Caller | `GpuEngine::step_slots_sampled` → `CudaBackend::launch_cooperative` |
| Launches / request | L + 127 (see §6.1) |
| Duration | **6.85–6.90 ms** at ctx 4096 (MEASURED FACT, whole step ≈ kernel; prior) |
| % of ITL | **~100%** of device ITL |
| Grid / block | 170 × 256 |
| Dtype | activations/weights **bf16**; flash partials **f32**; ids **i32** |

Optional second kernel (`plow_sample` in `runtime/nvidia/sample_sm120.cu`): **not** on the contract greedy path.

### 10.2 Device ops inside that kernel (Qwen T=1)

| op | kernel body | caller (emit) | launches/token (device packets) | shape (Qwen3-4B) | dtype |
|---|---|---|---|---|---|
| EMBED | `d_embed` `op_elementwise.cuh` | `emit_phase` | 1 | T=1, H=2560 | bf16 |
| RMSNORM | `d_rmsnorm` `op_norm.cuh` | first layer + final | few | H=2560, 1 row | bf16 |
| GEMV_QKV | `d_gemv_qkv` `op_gemm.cuh` | per layer | 36 | M=1, K=2560, N=q+k+v | bf16 |
| HEADNORM_ROPE | `d_headnorm_rope<128>` `op_norm.cuh` | per layer | 36 | hd=128, qk-norm on | bf16; writes K/V |
| FLASH_DECODE | `d_flash_decode<128, GF>` `op_attention.cuh` | per layer | 36 | GQA 32/8, nsplit≈6 | Q bf16, KV bf16, Opart f32 |
| FLASH_MERGE | `d_flash_merge<128>` | per layer (ns>1) | 36 | nsplit partials | f32→bf16 |
| GEMV | `d_gemv` | o_proj, down, lm_head | 36×2 + 1 | e.g. 2560×2560, 9728×2560, vocab×2560 | bf16 |
| ADD_NORM | `d_add_norm` `op_norm.cuh` | decode fuse_norm | 72 sites claimed in emit comments | H=2560 | bf16 |
| GEMV_GLU | `d_gemv_glu` | SwiGLU gate/up | 36 | inter×hidden | bf16 |
| ARGMAX / ARGMAX_FIN | `op_elementwise.cuh` | end of program | 1+1 | vocab | writes `in.ids` |

**Packet total 401** is **MEASURED FACT**. Per-op times and % **NOT MEASURED** (need `PLOW_NV_TRACE` or skeleton cubin or ncu — none this session).

**GEMM / FLASH_PREFILL:** **SOURCE FACT** — not present in the Qwen decode object; prefill buckets **trap**.

---

## 11. Ranked optimization hypotheses

Do **not** optimize in this agent. Expected benefit is order-of-magnitude where noted, not a promise.

### P0 — NVIDIA hd=128 prefill object (GEMM + FLASH_PREFILL)

- **Evidence:** SOURCE FACT trap/absent `_pf`; DERIVED TTFT ~27 s at L=4096 vs decode ITL 6.9 ms; Agent 2 fairness matrix “architectural”.
- **Source:** `runtime/nvidia/interp_sm120.cu` `PLOW_DOP_FLASH_PREFILL` / `GEMM_*`; emit `devgen` `emit_phase` `t>1`; loader `GpuEngine::load_prefill`.
- **Current cost:** **~4.81 ms × L** extra weight traffic vs one tiled prefill (DERIVED).
- **Expected benefit:** TTFT from **seconds → tens of ms** if kernels reach a respectable % of HBM/tensor-core (HYPOTHESIS, high — this is why vLLM is fast at prefill).
- **Risk:** wrong hd trap; Gemma `_pf` loaded by filename; occupancy vs Gemma hd256/512 object.
- **Recommended agent:** **Agent 4 (kernels)** + tiny **Agent 5** loader/mux once the cubin exists.

### P1 — Decode megakernel occupancy (1 → 2 blocks/SM)

- **Evidence:** SOURCE FACT `PLOW_NV_MINBLK=1`; comment “~21% of peak HBM”; fit `fixed_ms=1.70`.
- **Source:** `interp_sm120.cu` launch bounds; GEMV/FA register union.
- **Current cost:** **~1.7 ms/token** not explained by bytes (MEASURED intercept).
- **Expected benefit:** **partial** recovery of that 1.7 ms and/or higher achieved BW. Comment already warns spills may lose.
- **Risk:** spill traffic; correctness of cooperative grid vs `n_cu`.
- **Recommended agent:** **Agent 4**.

### P2 — Packet-gate tax (401 packets)

- **Evidence:** MEASURED 401 packets; AddNorm already deleted 72 vs split residual+norm; QKV fusion on bf16.
- **Source:** `devgen` `fuse_norm` / `fuse_qkv`; interpreter gate loop.
- **Current cost:** subset of 1.70 ms (HYPOTHESIS).
- **Expected benefit:** small vs P0; maybe 0.1–0.5 ms/tok if more fusion (HYPOTHESIS, low).
- **Risk:** occupancy regression from fatter arms; token mismatch.
- **Recommended agent:** **Agent 4** (bodies) / **Agent 5** only if host patching remains).

### P3 — Decode-loop prefill host sync (`PLOW_MULTISTEP` for prompt walk)

- **Evidence:** SOURCE FACT `gpu_prefill_advance` calls full `step_slots` (sync+D2H) per prompt token; `multi_step` already exists for decode.
- **Source:** `serve/mux.rs` `gpu_prefill_advance`; `GpuEngine::multi_step`.
- **Current cost:** host gap × L (NOT MEASURED; likely **≪** 4.81 ms×L).
- **Expected benefit:** low vs P0.
- **Risk:** pos/kvlen/counter re-arm races if stream order is mishandled.
- **Recommended agent:** **Agent 5**.

### P4 — Flash nsplit / GF fill on 170 SMs

- **Evidence:** SOURCE FACT ns=`ceil(n_cu/heads)=6` at ctx≤8192; 5090 has 170 SMs.
- **Source:** `devgen` nsplit; `d_flash_decode<128, GF>`.
- **Current cost:** part of KV term (0.36 ms @4k) + idle SMs during flash (HYPOTHESIS).
- **Expected benefit:** small at 4k (KV is 5% of step); larger at 32k+.
- **Risk:** merge partials scale with nsplit (Gemma measurements in emit comments).
- **Recommended agent:** **Agent 4**.

### P5 — ChatML template

- **Evidence:** SOURCE FACT `gpu_chat_prompt` has no `<|im_start|>` arm.
- **Current cost:** wrong wrap; **not** the 27 s prefill.
- **Expected benefit:** fairness vs vLLM token counts; maybe ±tens of prompt tokens.
- **Risk:** changes tokenized L; **must re-run Agent 2 argv**, not hide in the bench.
- **Recommended agent:** product/runtime (**Agent 5**), then **re-baseline**.

### P6 — Prefix cache / VMM

- **Evidence:** contract both off; random dataset `random-prefix-len=0`.
- **Current cost:** 0 vs this workload.
- **Recommended agent:** none for this contract cell.

### Explicitly not recommended (evidence against)

| idea | why |
|---|---|
| CUDA graphs on plow decode | already 1 launch/token |
| `FA_BKV_D128=64` | AMD campaign **slower** (Agent 1); different GPU but do not revive blindly |
| Disable vLLM graphs “for fairness” | forbidden by Agent 2 |
| Quote gfx950 2.9× as this GPU’s gap | different prefill algorithm |

---

## 12. Mapping metrics → this path

| metric | what actually dominates on NVIDIA Qwen today | tag |
|---|---|---|
| **TTFT** | decode-loop prefill = **L × ~6.5–7.1 ms** | DERIVED |
| **Prefill latency** | same as TTFT minus HTTP/template (still ≈ T_prefill) | DERIVED |
| **ITL / TPOT / decode** | one interp launch; **~6.87 ms @4k** | MEASURED (prior) |
| **E2E** | TTFT + 127×ITL ≈ **T_prefill + 0.87 s** at L=4096 | DERIVED |

Worked E2E (device+sync only, **not** HTTP):

| L | DERIVED E2E |
|---:|---|
| 1024 | ~6.71 s + 127×6.60 ms ≈ **7.55 s** |
| 4096 | ~27.4 s + 127×6.87 ms ≈ **28.3 s** |
| 8192 | ~56.3 s + 127×7.13 ms ≈ **57.2 s** |

---

## 13. Follow-up to produce ncu-quality rows

On an exclusive 5090 with weights + CUDA `plowrt`:

1. Log line: `no prefill object for sm_120 — decode-only prompt consumption`.
2. `PLOW_STEP_TIME=1` during a conc=1 contract pass (or `step_bench` as extra).
3. `PLOW_TTFT_LOG=1` for host phases.
4. If ncu admin can be lifted: one request at L=1024, report `interp_sm120` duration vs memcpy/memset.
5. Do **not** load Gemma `_pf` to “get a prefill profile” — that is a different model.

---

## 14. Verification of this document

- Path: `docs/agent3-profile.md`.
- Production inference/benchmark code: **not modified**.
- Canonical HTTP A/B: **not run** (blockers listed in §0).
- Decode step table: **MEASURED FACT (prior, this GPU)** cited from committed `qsim.rs` / `mdq.rs`.
- Prefill/TTFT/E2E seconds: **DERIVED**, labeled as such.
