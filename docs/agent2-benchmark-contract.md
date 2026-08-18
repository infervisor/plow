# Agent 2 — Qwen3-4B benchmark contract

> **Immutable for Agents 3–6.** Do not change workload, client, metric
> arithmetic, or timing boundaries to make plow look better.
>
> There is **no Qwen ASR / speech encoder** in this tree (Agent 1). The
> comparable production path is the **Qwen3 dense GQA decoder**
> (`model_type: "qwen3"`). Campaign name in git: “qwen asr” → **Qwen3-4B
> text-to-text**.

Status of **this Agent 2 checkout** (2026-08-18):

- Hardware present: **1× NVIDIA GeForce RTX 5090** (sm_120a), driver 580.142.
- **No AMD GPU** (`/dev/kfd` absent). Historical gfx950 / MI350X numbers are
  **out of contract**.
- Live A/B was **not executed**: no Qwen3-4B weights, no `plowrt` binary, no
  CUDA vLLM server, `nix develop` blocked in this agent sandbox. See
  `docs/baseline-results.md`.

The contract below is still the canonical comparison. Fill
`docs/baseline-results.md` on a box that can actually run it. Do not
substitute `amd-bench`, `bench_speed.sh`, or twoengine `client.py` numbers.

---

## 1. Canonical comparison

**Same client, same protocol, two servers.**

| side | server | client |
|---|---|---|
| OUR IMPLEMENTATION | `plowrt serve` OpenAI chat | `vllm bench serve --backend openai-chat` |
| VLLM | `vllm serve` OpenAI chat | **identical** client argv |

plowrt has **no** `/v1/completions` (`serve/mod.rs`). Mixing `--backend openai`
(vLLM) with `--backend openai-chat` (plow) compares different TTFT events.
Forbidden.

Do **not** table these against the canonical client:

| instrument | why it is not this contract |
|---|---|
| `plowrt amd-bench` | device loop only; no HTTP, detok, template |
| `scripts/bench_speed.sh` | different client; header forbids vLLM pairing |
| `scripts/twoengine/client.py` | non-streaming `max_tokens=1` JSON buffer |
| `runtime/tests/qwen3_sm120_chat.cu` | decode-loop correctness, not served TTFT |
| in-process `vllm.LLM` | no HTTP; historical 0.25.1 files |

Allowed extra (not headline): `PLOW_TTFT_LOG=1` server breakdown, conc=1 only
(`crates/plowrt/src/obs/ttft.rs`).

---

## 2. Environment (pin at run time)

Must use the flake. Default shell for plow; `.#vllm` for the **client only**.

```bash
nix develop                 # plow-dev: cargo, nvcc/hipcc, LD_LIBRARY_PATH
nix develop .#vllm          # vLLM 0.27.0 HTTP client; not a GPU server
```

| item | this checkout | record at every run |
|---|---|---|
| nixpkgs | `flake.lock` `49a4bd0573c376468dd7996ddb6f9fa31d8c4d97` | `nix flake metadata` |
| Nix | 2.35.2 (`/nix/var/nix/profiles/default`) | `nix --version` |
| CUDA (host) | 13.0 (`CUDA_VERSION 13000`, `/usr/local/cuda`) | `$PLOW_NVCC --version` inside `nix develop` |
| CUDA (flake) | `pkgs.cudaPackages.cudatoolkit` via `PLOW_NVCC` | same |
| ROCm (flake) | TheRock **7.14.0** (`PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix`) | unused on this NVIDIA box |
| Python (client) | nix `.#vllm` | `python3 --version` |
| PyTorch (host conda) | 2.9.0+cu130 — **not** the plow serve path | only if used to serve vLLM |
| GPU | NVIDIA GeForce RTX 5090, **32 GiB**, PCI `0000:41:00.0` | `nvidia-smi` |
| GPU count (visible) | **1** (`/dev/nvidia2` only) | `nvidia-smi -L` |
| Driver | 580.142 Open Kernel Module | `nvidia-smi` |
| Power | sysfs `power_state=D0`; no clock files | `nvidia-smi -q -d CLOCK,POWER` |
| MIG | off (RTX 5090; empty `/dev/nvidia-caps`) | `nvidia-smi -L` |
| Clocks | **not locked** | persistence + application clocks if available |

**Same GPU for both engines.** On this box that is CUDA device index 0
(`nvidia2`). Set `CUDA_VISIBLE_DEVICES` identically. Do not run under a
busy `nvidia_uvm` (this container had **28** uvm users at inspection).

**Refuse the run** if plowrt logs `CPU reference backend active` (correct
tokens, fictional timings). NVIDIA: require CUDA engine selected, not CPU.

Relevant env (leave unset unless documented):

| var | contract |
|---|---|
| `CUDA_VISIBLE_DEVICES` | same single index both sides |
| `PLOW_VMM_PREFIX` | **off** (default false) |
| `PLOW_PREFIX_CACHE` | **off** (default false) |
| `PLOW_TTFT_LOG` | `1` on one conc=1 diagnostic pass only |
| `PLOW_PF_BATCH` | default (serialized prefill) |
| `HF_HUB_OFFLINE` | `1` once weights are local |

---

## 3. Model

| field | value |
|---|---|
| HF id | `Qwen/Qwen3-4B` |
| Architecture | dense GQA decoder; `head_dim=128`, heads 32/8, hidden 2560, 36 layers, SwiGLU, q/k RMSNorm, tied `lm_head` |
| Dtype | **bfloat16** weights and activations |
| Quantization | **none** (vLLM fp8 is slower than vLLM bf16 on 4B — `perf-data/vllm-fp8-baseline.md`) |
| Revision | local snapshot SHA; write it into `baseline-results.md` |
| Context compile | plow `--max-ctx` ≥ 16384; vLLM `--max-model-len 16384` |
| Native rope | 40960; **no** yarn/linear scale |

plow assets: `plowc --hf-dir <ckpt> --emit devblob` for **this** GPU
(`--arch sm_120` / the NVIDIA emit path used on sm_120). Tokenizer =
checkpoint `tokenizer.json` on **both** client `--tokenizer` and server.

---

## 4. Workload (immutable points)

| knob | value | why |
|---|---|---|
| Dataset | `vllm bench serve --dataset-name random` | same token factory both sides |
| Input lengths | **1024, 4096, 8192** | script default + campaign 4k/8k |
| Output length | **128** | harness default |
| `--random-range-ratio` | **0** | exact length, not a window |
| Shared prefix | **0** (`--random-prefix-len` unset/0) | plow prefix cache off |
| Concurrency | **1** | latency contract; AMD batch=1 would queue anyway |
| Batch (plow packet) | compile `PLOW_DECODE_BATCH=1` | like-for-like with conc=1 |
| Prompts / point | **32** | NPROMPT=8 poisons the mean (see §10) |
| Warmup | **4** discarded requests | `--num-warmups 4` |
| Seed | **0** | `--seed 0` |
| Request rate | `inf` | closed loop, conc=1 |

One cell = one `(input_len, conc=1)` line. Run **all three** input lengths.
Do not drop 1024 or 8192 to hide a gap.

**Validity gate:** `Total generated tokens == 32 × 128 = 4096` per cell.
`Successful requests` **alone is not a gate** — vLLM counts some rejects as
success (`scripts/bench_plowrt_serve.sh`).

Coherence **before** timing, both servers: “What is the capital of France?”
greedy, must contain `paris`.

---

## 5. Canonical commands

### 5.1 Client (both servers)

Use the flake client, **not** the ROCm docker image (no docker in this
container; that image is AMD).

```bash
# env: PORT MODEL TOKZ L TAG OUT
nix develop .#vllm --command vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions \
  --model "${MODEL}" \
  --tokenizer "${TOKZ}" \
  --dataset-name random \
  --random-input-len "${L}" \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency 1 \
  --num-prompts 32 \
  --num-warmups 4 \
  --ignore-eos \
  --temperature 0 \
  --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result --save-detailed \
  --result-dir "${OUT}" \
  --result-filename "${TAG}_in${L}_c1.json"
```

`L ∈ {1024, 4096, 8192}`. Same `TOKZ`, `seed`, flags on plow and vLLM.

Wrappers `scripts/bench_plowrt_serve.sh` / `scripts/bench_vllm_chat.sh` are
the same **client idea** but default `NPROMPT=8`, no `--ignore-eos`, no
percentiles, and they call `sudo -n docker` + ROCm image. **Override or do
not use them on this NVIDIA box.** Contract = the argv above.

### 5.2 OUR IMPLEMENTATION (plowrt)

```bash
# private binary — a concurrent `cargo build -p plowrt` without cuda/hsa
# silently replaces target/release with the CPU interpreter
PLOWRT_BIN=/path/to/private/plowrt

grep -aq libcuda "$PLOWRT_BIN" || exit 1   # NVIDIA analog of the HSA gate

nix develop -c "$PLOWRT_BIN" serve \
  --assets "$ASSETS" \
  --port 8101
```

Build: `nix develop -c cargo build --release -p plowrt --features cuda`.

NVIDIA Qwen prefill object: default `interp_sm120_pf.cubin` implements
FLASH_PREFILL for **hd 256/512 only**; hd 128 **`__trap()`s**
(`runtime/nvidia/interp_sm120.cu`). `GpuEngine` loads `_pf` when present and
`mux` then calls `prefill_chunk` (`has_prefill()`). For Qwen3-4B:

- **If `_pf` cubin is next to the asset → launch traps.** Not a result.
- **If `_pf` is absent → decode-loop prompt consumption** (`gpu_prefill_advance`
  else branch): one T=1 launch per prompt token. Correct tokens, **O(n)
  launches**. That **is** the current NVIDIA Qwen serve path. Do not hide it.

Record in the result file which path ran (`"prefill object loaded"` vs
`"no prefill object … decode-only prompt consumption"`).

### 5.3 VLLM

Nix `.#vllm` is **client-only** (`dontBuild`; no CUDA engine). Serve with a
CUDA vLLM that can load Qwen3-4B on sm_120, **same weights**.

```bash
CUDA_VISIBLE_DEVICES=<same> vllm serve "${CKPT}" \
  --served-model-name Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --port 8102
```

| vLLM knob | contract |
|---|---|
| Version | record `vllm.__version__` (target: CUDA build near **0.27.x** to match the client) |
| Quantization | none |
| Attention backend | whatever vLLM selects for Qwen3 GQA on sm_120; **log it** (FLASHINFER / FLASHATTN / TRITON / …) |
| Compilation | default `torch.compile` / cudagraph path; **not** `--enforce-eager` |
| CUDA graphs | ON (vLLM default). plow decode is already 1 launch/token; graphs are a vLLM advantage — **keep them**, do not disable to “fair” plow |
| TP | 1 |
| Prefix cache | **off** |
| Chunked prefill budget | `--max-num-batched-tokens 8192` (existing ROCm harness parity) |

ROCm image `rocm/vllm:…vllm_0.23.0` is **not** runnable here. If a later
agent moves to gfx950, that image is the historical AMD server; **re-pin
and re-measure**. Do not mix AMD docker numbers with this RTX 5090 contract.

---

## 6. Sampling / decoding

| field | value | code |
|---|---|---|
| Temperature | **0** | client `--temperature 0` → plow `temp <= 0` keeps device `ARGMAX_FIN` |
| top_p / top_k | unused (greedy) | |
| `ignore_eos` | **true** | synthetic random must emit exactly 128 tokens (`GenParams::ignore_eos`) |
| Stop | length cap only | |
| Stream | **true** (client) | TTFT is first SSE `choices` chunk |

plow default `SamplingParams.temperature` is 1.0 if the request omits it.
The client **must** send `temperature: 0`.

---

## 7. Metric definitions (from code, not folklore)

Client clock: `time.perf_counter()` in vLLM
`benchmarks/backend_request_func.py` (openai-chat). **CPU wall time around
HTTP.** The server does **not** export CUDA events to this client.

plow SSE (`serve/chat.rs` `sse_response`): the `role: assistant` delta
**rides the first generated-token chunk**. It is not a leading empty frame.
Comment in that function: vLLM stamps TTFT on the first chunk with a
`choices` array **regardless of content**. A role-only frame at arrival made
plow TTFT = one RTT (measured 7.1 ms vs 1322 ms first token). That bug is
fixed. Do not reintroduce a leading role frame.

### 7.1 TTFT

| | |
|---|---|
| Start | client `perf_counter` immediately before the HTTP POST |
| End | first SSE `data:` JSON whose `choices` is non-empty |
| Includes | localhost HTTP, axum accept/JSON, **chat template**, HF tokenize, mux queue, `begin_slot`, **entire prompt consumption** (prefill kernel **or** decode-loop), D2H of first token, detok, SSE serialize |
| Excludes | tokens after the first; client-side tokenizer of later text |
| Server diagnostic | `t_arrive` at handler entry → first SSE frame (`obs/ttft.rs`). Axum accept/JSON is UNACCOUNTED vs the client (client includes them) |

**Not pure device prefill.** Conc=1 TTFT ≈ prefill + first-token sample +
host. `PLOW_TTFT_LOG` phase `prefill TOTAL (engine thread)` is closer to
device+HtoD+drain.

### 7.2 Prefill latency

**Not a first-class `vllm bench serve` metric.**

| derivation | valid? |
|---|---|
| TTFT at this workload | **proxy only**; includes HTTP + template + first token |
| `max_tokens=1` non-streaming JSON (`twoengine/client.py`) | includes full response buffer; **not** this contract |
| `PLOW_TTFT_LOG` `PREFILL` | plow-only; conc=1; after handler entry |
| `amd-bench` / CUDA events | not HTTP; not comparable to vLLM TTFT |

Do not label TTFT as “prefill ms” in Agent 3–6 tables. Historical
`perf-data/plow-vs-vllm-baseline.md` did that for vLLM docker — **flagged
there**, forbidden here.

### 7.3 Inter-token latency (ITL)

For each successful request, ITL samples are

```
t[i] − t[i-1]   for i = 2..N_chunks_with_choices
```

**First generated token is not an ITL sample** (it closed TTFT).
**First decode step is included** — it is the gap from token 1 → token 2.

Mean / median / P50 / P90 / P95 / P99 are over the **flattened** list of all
ITLs in the cell (vLLM `all_itls += output.itl`), not per-request means.

Empty-delta SSE frames (partial UTF-8) still count as chunks on plow. Rare
at greedy English; if `output_lens` ≠ 128, the cell is invalid.

### 7.4 Decode latency / TPOT

vLLM TPOT per request (output_len > 1):

```
(e2e_latency − ttft) / (output_len − 1)
```

Mean/median/percentiles over **per-request** TPOT. With `ignore_eos` and
fixed 128 tokens this is ~ mean ITL but **not identical** (ITL is pooled
gaps; TPOT is mean of per-request averages).

Does **not** include prefill.

### 7.5 End-to-end latency (E2EL)

Start = same as TTFT. End = client time at `[DONE]` (full stream).
Includes prefill + all decode + last SSE. Report P50/P90/P95/P99.

### 7.6 Throughput

```
output_tok/s = (sum completion tokens) / (cell wall clock)
request/s    = (successful requests)   / (cell wall clock)
```

Wall clock = client `benchmark_duration` covering the **measured** 32
prompts, **excluding** the 4 warmups. Do not use `1000/mean_tpot` as
throughput (that ignores prefill and concurrency).

### 7.7 Required stats

From `--save-detailed` JSON + printed summary, per cell, per engine:

TTFT, TPOT, ITL, E2EL: **mean, std, min, max, median, P50, P90, P95, P99**.

Also: `completed`, `failed`, `total_input_tokens`, `total_output_tokens`,
`output_tok/s`, `req/s`.

P50 may equal median. Still record both if the client prints both.

---

## 8. Synchronization (measurement honesty)

### 8.1 What the HTTP client sees

The client times **CPU** around asynchronous HTTP. That is valid **iff** the
server does not send the first token before the GPU work that produced it
has completed.

### 8.2 plow NVIDIA (`GpuEngine::step_slots_sampled`)

Patch, H2D, launch, D2H are enqueued on one stream and retired by **one**
`cuStreamSynchronize` (`exec/gpu.rs`). Token readback is after that sync.
Prefill chunks likewise `stream_synchronize` before `download` of `in.ids`.

CUDA events exist for **optional load/profile overlays**, not the HTTP
clock.

### 8.3 plow AMD (not this box)

AQL enqueue of all segments, **one** `drain` per chunk, then `read_sampled`.
Same property: SSE after device completion.

### 8.4 Misleading timings (document, do not “fix” by changing work)

| bug | effect | contract response |
|---|---|---|
| CPU interpreter fallback | fluent, ~100× too fast | abort |
| No warmup, NPROMPT=8 | one cold request owns the mean (measured 1880 vs 573 ms TTFT) | warmup 4, n=32 |
| Leading role SSE (fixed) | TTFT ≈ RTT; prefill lands in first ITL | keep current `sse_response` |
| Chat template mismatch | different tokenized prompts | §9 — do not hide |
| NVIDIA Qwen `_pf` cubin | trap or Gemma hd256 path | remove `_pf` or abort |
| NVIDIA decode-loop prefill | O(n) vs vLLM O(n/C) | **report as the implementation**, not a client bug |
| Prefix cache on vLLM only | feature gap as kernel gap | both off |
| `Successful requests` with 0 gen tokens | fake throughput | require gen_toks |
| `--backend openai` vs `openai-chat` | different TTFT event | chat only |
| Unlocked clocks / neighbor GPU load | DVFS noise | record clocks; exclusive GPU |
| `PLOW_TTFT_LOG` at conc>1 | global timers mix requests | conc=1 only |

**No `torch.cuda.synchronize()` on the client.** None needed: work is on the
server, and the server already syncs before the byte the client timestamps.

---

## 9. Fairness matrix

| axis | plow | vLLM | like-for-like? |
|---|---|---|---|
| Weights | same HF snapshot | same | **must** |
| Precision | bf16 | bf16 | yes |
| GPU | RTX 5090 #2 | same | **must** |
| Input factory | random, seed 0, range 0 | same | yes |
| Chat template | **Gemma-4 markers** (`gpu_chat_prompt` falls through; no `<|im_start|>` arm) | checkpoint ChatML | **NO** |
| Tokenized prompt length | Gemma wrap + user text | ChatML wrap + user text | **record both `total_input_tokens`** |
| Output | 128, ignore_eos | same | yes if gate holds |
| Batch / conc | 1 / 1 | 1 / 1 | yes |
| Decoding | greedy argmax | greedy | yes if temp=0 |
| Prefix cache | off | `--no-enable-prefix-caching` | yes |
| Warmup / n | 4 / 32 | 4 / 32 | yes |
| TTFT boundary | first `choices` SSE = first token | first `choices` SSE | yes with current plow SSE |
| Prefill algorithm | sm_120: **decode-loop** (or trap) | vLLM flash/GEMM prefill | **NO — architectural** |
| CUDA graphs | N/A (1 interp launch) | ON | vLLM extra; keep |
| HTTP + detok | included | included | yes |

**Chat template** is an implementation defect, not a bench knob. Agents 3–6
must not “fix” it inside the benchmark (e.g. by feeding pre-templated
strings that vLLM will wrap again). If a later **product** change adds
ChatML, re-measure both sides; do not splice old TTFT.

**NVIDIA prefill** is the other real gap. A 4k TTFT ratio on this GPU
measures decode-loop prefill vs vLLM prefill until an hd=128 `_pf` object
exists. That is still the correct “ours vs vLLM” number for **this**
implementation. Do not switch plow to `amd-bench` to look closer.

---

## 10. Script defaults vs this contract

| | `bench_plowrt_serve.sh` default | this contract |
|---|---|---|
| NPROMPT | 8 | **32** |
| warmup | unset (0) | **4** |
| `--ignore-eos` | omitted | **on** |
| `--temperature 0` | omitted | **on** |
| `--random-range-ratio 0` | omitted | **on** |
| percentiles | parse mean/median/p99 ITL only | **50,90,95,99** on ttft,tpot,itl,e2el |
| `--save-detailed` | no | **yes** |
| docker ROCm client | yes | **no** (use `nix develop .#vllm`) |

Leaving NPROMPT=8 / no warmup is a **known measurement bug** (script’s own
header). Overriding via argv is a measurement-only correction. Scripts were
**not** edited in this Agent 2 pass.

---

## 11. What Agents 3–6 may / must not do

**Must not**

- Change L, OUTLEN, conc, n, warmup, seed, dtype, model id, ignore_eos.
- Enable prefix cache on one engine.
- Disable vLLM graphs “for fairness”.
- Quote gfx950 campaign ratios as this baseline.
- Optimize kernels **and** retune the bench in the same change.
- Drop the Paris gate or the gen_toks gate.

**May**

- Add `PLOW_TTFT_LOG` diagnostic rows (extra, not replacement).
- Record GPU clocks / power beside the same cells.
- After a **documented** ChatML or hd=128-prefill product change, re-run
  **this same argv** and add a new results file (do not rewrite this
  contract’s workload).

---

## 12. Reproduce checklist

1. Exclusive RTX 5090; `nvidia-smi` clocks/power snapshot.
2. `nix develop` versions: nvcc, python (vllm shell), plowrt git SHA.
3. Qwen3-4B snapshot SHA; bf16; no quant.
4. Private `plowrt` with CUDA; log shows CUDA engine, not CPU.
5. Prefill path logged (decode-only vs `_pf`).
6. Paris gate both servers.
7. Client argv §5.1 for L=1024,4096,8192 both base URLs.
8. JSON: `completed==32`, `total_output_tokens==4096`, empty `errors`.
9. Fill `docs/baseline-results.md` with raw + mean/median/p50/p90/p95/p99/std/min/max.
