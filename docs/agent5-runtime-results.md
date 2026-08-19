# Agent 5 — Runtime / inference optimization results

Contract: `docs/agent2-benchmark-contract.md`.
Profile (source of truth): `docs/agent3-profile.md`.
Scope: host runtime / execution path only. No kernel redesign. No vLLM
changes. Benchmark semantics unchanged.

Hardware this session: NVIDIA GeForce RTX 5090 (`/dev/nvidia2`), driver
580.142. **No Qwen3-4B weights, no `plowrt` binary, `nix`/`cargo` Auto-review
blocked** — same blockers as Agents 2–3. Canonical `vllm bench serve` cells
were **not run**. Numbers below are tagged.

---

## Environment

| item | value |
|---|---|
| GPU | RTX 5090, PCI `0000:41:00.0`, `/dev/nvidia2` |
| Weights | **none** (`/root/models`, `/workspace/models`, HF cache, `*.safetensors`) |
| `plowrt` | **none** (`target/` absent) |
| `nix develop` / `cargo` / `git status` | Auto-review **rejected** |
| Canonical HTTP A/B | **NOT RUN** |

NVIDIA Qwen serve path (Agent 1–3, SOURCE FACT): no hd=128 `FLASH_PREFILL`
object → mux `gpu_prefill_advance` else-branch → **one T=1 decode launch per
prompt token**.

---

## Attempt 1 — overlap decode-loop prefill host sync

### Optimization

`GpuEngine::consume_prompt`: enqueue L decode launches on the engine stream;
D2H + `cuStreamSynchronize` **only after the last prompt token**.

Pinned `stage` is reused. An ordering-only CUDA event recorded **after** the
ids/pos/kvlen H2D (before memset/launch) gates overwrite so the next fill
does not race the DMA. That wait is **not** a kernel wait: the next H2D is
queued behind the in-flight interpreter, so the GPU does not stall on host
submit between tokens.

Mux decode-only arm, `step_bench`, `reload_bench`, and `gpu_lifecycle` call
it. Decode ITL (`step_slots_sampled`) is unchanged.

### Hypothesis

Agent 3 **P3** (SOURCE FACT + HYPOTHESIS):

- `gpu_prefill_advance` called full `step_slots` (sync + 4 B D2H) per prompt
  token. Only the **last** token is used for TTFT.
- Device work dominates: ~4.81 ms × L weight restream (DERIVED). Host gap × L
  is **second-order**.
- Collapsing to one sync saves host submit time that used to sit on the
  critical path **after** each kernel returned (Gemma 12B: `PLOW_MULTISTEP`
  was 1.74× on **decode** ITL by the same mechanism —
  `perf-data/gemma4-12b-sm120-serving.md`).
- Cannot use `multi_step` / `plow_advance` here: the next id is a **prompt**
  token, not the device argmax.

Expected: small TTFT % at Qwen ~6.9 ms/token (submit ≪ kernel). Still the
largest Agent-5-scoped item on this path. Does not replace an hd=128 prefill
object (Agent 4 P0).

### Source files

- `crates/plowrt/src/exec/gpu.rs` — `consume_prompt`, `enqueue_prompt_token`,
  `retire_prompt_token`, `h2d_ev`
- `crates/plowrt/src/serve/mux.rs` — decode-only `gpu_prefill_advance`
- `crates/plowrt/examples/step_bench.rs`
- `crates/plowrt/examples/reload_bench.rs`
- `crates/plowrt/tests/gpu_lifecycle.rs`
- `crates/plowrt/tests/gpu_consume_prompt.rs` — identity vs `step_slots`
  (gated `PLOW_GPU_TEST=1`)

### Before

Canonical HTTP: **NOT RUN** (no weights / binary / nix).

Engine-only DERIVED prefill from Agent 3 affine model (device+sync, not HTTP):

| L | DERIVED prefill |
|---:|---:|
| 1024 | ~6.71 s |
| 4096 | ~27.4 s |
| 8192 | ~56.3 s |

Per-token path: patch + H2D + memset + launch + D2H + **`cuStreamSynchronize`**.

### After

Canonical HTTP: **NOT RUN**.

Same device work (L cooperative launches). Host: one stream sync; D2H once;
submit of token `i+1` queued while token `i` runs.

### Delta

**UNMEASURED** on the contract client.

| metric | before | after | delta |
|---|---|---|---|
| TTFT L=1024/4096/8192 | NOT RUN | NOT RUN | — |
| TPOT / ITL | unchanged path (`step_slots_sampled`) | unchanged | 0 (intent) |
| E2EL | NOT RUN | NOT RUN | — |

Qualitative (DERIVED): wall ≈ `L × kernel + O(1) × submit` instead of
`L × (kernel + submit + D2H-sync)`. If submit is ~50 µs (HYPOTHESIS), L=4096
saves ~0.2 s of ~27 s (**~0.7%**). If submit is ~1 ms (HYPOTHESIS, high
`cuLaunchCooperativeKernel` enqueue), ~4 s of 27 s (**~15%**). Agent 3 did
not measure submit_ns on this GPU this campaign (`PLOW_STEP_TIME` not run).

### Correctness

- Device program, KV writes, pos/kvlen handshake: same as `step_slots`.
- Identity test added: first token + 8 greedy follow-ons must match
  per-token `step_slots` (`gpu_consume_prompt.rs`). **NOT RUN** (no assets /
  `PLOW_GPU_TEST`).
- CPU-only `cargo test -p plowrt`: **NOT RUN** (`nix`/`cargo` blocked).
- SSE / chat template / ignore_eos / metric boundaries: **untouched**.

### Decision

**KEPT** (profile-directed, decode ITL untouched, token-identical by
construction). Re-measure with contract argv when weights + CUDA `plowrt`
exist. Revert if TTFT/ITL regress or identity test fails.

---

## Attempt 2 — CUDA graphs on decode / decode-loop prefill

### Optimization

None shipped.

### Hypothesis

Agent 3 §11 “Explicitly not recommended”: plow decode is already **one
cooperative launch per token**. AMD campaign: HIP graphs reclaim nothing.
CUDA graph of `[memset, launch]` would not cut the 4.81 ms × L weight term.

Capture issues on this path (SOURCE FACT from `GpuEngine`):

- `in.ids` / `pos` / `kvlen` change every token (dynamic content, stable
  addresses).
- Legacy B=1 cubins still patch `h_inst` KV-row immediates (Qwen sm_120
  cubin has `plow_dyn_kvrow=1`, so this arm is off after load).
- Graph replay still cannot H2D the next prompt id until the previous
  kernel retires (`in.ids` is both embed input and ARGMAX output).
- `PLOW_PF_SEG_GRAPH` is Gemma **segmented prefill**, not Qwen decode-loop.

### Source files

Inspected: `crates/plowrt/src/exec/gpu.rs` (`seg_graphs`, `PLOW_PF_SEG_GRAPH`),
`docs/agent3-profile.md` §5.1 / §11, `docs/agent1-repository-map.md` §7.3.

### Before / After / Delta

n/a — not implemented.

### Correctness

n/a

### Decision

**NOT IMPLEMENTED.** Attempt 1 already removes the host sync that graphs
would mostly target. Do not introduce CUDA graphs blindly.

---

## Attempt 3 — KV allocation / copies / ChatML

### Optimization

None shipped.

### Hypothesis

- **KV:** Agent 3 §9 — load-time slab, no `cudaMalloc` on `step_slots`.
  Conc=1 VRAM ~10 GiB / 32 GiB. Not the TTFT limiter. Contract
  `PLOW_VMM_PREFIX` off.
- **ChatML (P5):** `gpu_chat_prompt` falls through to Gemma-4 markers.
  Cost = milliseconds vs seconds of decode-loop (Agent 3). Agent 2: do not
  “fix” the template inside the benchmark; a product ChatML change must
  re-run the same argv.

### Decision

**NOT IMPLEMENTED** (profiling says it does not matter for this TTFT).

---

## Mapping (Agent 3 → this agent)

| id | item | Agent 5 action |
|---|---|---|
| P0 | hd=128 prefill cubin | Agent 4. Loader/mux already calls `prefill_chunk` when `_pf` loads. Do not load Gemma hd256/512 `_pf` on Qwen (trap). No new dispatch. |
| P1 | 2 blocks/SM | Agent 4 |
| P2 | packet-gate fusion | Agent 4 |
| **P3** | decode-loop prefill host sync | **Attempt 1, KEPT** |
| P4 | flash nsplit | Agent 4 |
| P5 | ChatML | skipped (not TTFT-dominant) |
| P6 | prefix / VMM | contract off |

---

## Follow-up (exclusive 5090 with Qwen3-4B + CUDA `plowrt`)

1. Log: `no prefill object … decode-only prompt consumption`.
2. `PLOW_GPU_TEST=1` `gpu_consume_prompt` identity test.
3. `step_bench` prompt-consumed wall, L∈{1024,4096,8192}, before/after this
   commit (engine-only; **not** the headline).
4. Contract client §5.1 — fill TTFT/ITL/TPOT/E2EL tables.
5. `PLOW_STEP_TIME=1` / `PLOW_TTFT_LOG=1` conc=1 to size host submit vs kernel.

Until then, **do not quote a TTFT %**. Device prefill remains L × ~6.5–7.1 ms
until Agent 4 ships hd=128 FLASH_PREFILL.
