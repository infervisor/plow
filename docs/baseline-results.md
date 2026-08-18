# Baseline results — Qwen3-4B vs vLLM

Contract: `docs/agent2-benchmark-contract.md` (immutable).

**This file contains no invented timings.** The canonical A/B defined in the
contract was **not executed** in this Agent 2 session. Sections
**OUR IMPLEMENTATION** and **VLLM** are therefore empty of latency numbers.

Date of inspection: **2026-08-18**.

---

## 1. Run status

| cell | status |
|---|---|
| OUR IMPLEMENTATION (plowrt HTTP) | **NOT RUN** |
| VLLM (served HTTP, same client) | **NOT RUN** |
| Coherence gate | **NOT RUN** |
| Warmup 4 + 32 prompts × {1024,4096,8192} | **NOT RUN** |

### Blockers (all must be cleared before a number is a result)

1. **No Qwen3-4B weights.** Missing `/root/models/Qwen3-4B`,
   `/workspace/models`, `$HOME/.cache/huggingface`, any `*.safetensors`.
2. **No `plowrt` binary.** `target/release/plowrt` does not exist.
3. **No CUDA vLLM server.** No `/workspace/vllm*`, no
   `rocm7-bench-venv`, no `vllm` in conda site-packages. Nix `.#vllm` is
   **client-only** (0.27.0, `dontBuild`).
4. **No docker CLI** (process is already in a container). ROCm client image
   used by `bench_plowrt_serve.sh` cannot be started here.
5. **`nix develop` / `nvidia-smi` were Auto-review blocked** in this agent
   sandbox. Environment below is from files and sysfs, not a live
   `nvidia-smi -q`.
6. **`nvidia_uvm` had 28 users** at inspection — GPU not exclusive.

Do not copy Appendix A into Agent 3–6 headline tables. Those numbers are a
**different GPU, different vLLM version, mixed instruments**.

---

## 2. Environment recorded (this box)

| item | value | how obtained |
|---|---|---|
| Kernel | Linux 6.8.0-65-generic | `/proc/version` |
| OS | Ubuntu 22.04.5 LTS | `/etc/os-release` |
| Nix | 2.35.2 | profile store path |
| nixpkgs | `49a4bd0573c376468dd7996ddb6f9fa31d8c4d97` | `flake.lock` |
| CUDA (host) | **13.0** (`CUDA_VERSION 13000`) | `/usr/local/cuda/include/cuda.h` |
| nvcc | present `/usr/local/cuda/bin/nvcc` | `ls` |
| Driver | **580.142** Open Kernel Module | `/proc/driver/nvidia/version` |
| GPU (visible) | **1× NVIDIA GeForce RTX 5090** | `/proc/driver/nvidia/gpus/*/information` |
| PCI / node | `0000:41:00.0` minor 2, `/dev/nvidia2` | sysfs + `/dev` |
| VRAM | **32 GiB** (BAR1) | GPU information |
| Other host GPUs | 2 further RTX 5090 (`c1:00.0`, `81:00.0`) **not passed through** | same |
| MIG | off | `/dev/nvidia-caps` empty; no MIG params |
| Power | `power_state=D0`, runtime active | `/sys/class/drm/card3/device/` |
| GPU clocks | **not read** (`nvidia-smi` blocked; no `*clock*` sysfs) | — |
| ROCm | **absent** (no `/dev/kfd`, no `/opt/rocm*`) | `ls` |
| Host Python | 3.11.14 (`/opt/conda`) | conda-meta |
| Host PyTorch | 2.9.0+cu130 | site-packages |
| `CUDA_VISIBLE_DEVICES` / `PLOW_*` | **unset** in this process | `/proc/self/environ` |
| plowrt | missing | `ls target/release/plowrt` |

Contract GPU = this visible 5090. Historical Qwen campaign GPU = **AMD
gfx950 / MI350X** — not this machine.

---

## 3. OUR IMPLEMENTATION

**No measurements.**

Expected server (when runnable):

```text
nix develop -c $PLOWRT_BIN serve --assets <qwen3-4b-sm120-assets> --port 8101
```

Expected client: contract §5.1 with `TAG=plow`, `PORT=8101`.

Required log lines before trusting any row:

- CUDA engine selected (not `CPU reference backend active`)
- either `no prefill object for sm_120 — decode-only prompt consumption`
  **or** a verified hd=128 FLASH_PREFILL object (does not exist in default
  `interp_sm120_pf.cubin` today)
- coherence gate PASS

| L | n | warmup | TTFT mean/med/p50/p90/p95/p99/std/min/max | TPOT … | ITL … | E2EL … | out tok/s |
|---|---|---|---|---|---|---|---|
| 1024 | — | — | **NOT RUN** | | | | |
| 4096 | — | — | **NOT RUN** | | | | |
| 8192 | — | — | **NOT RUN** | | | | |

Raw JSON: none.

---

## 4. VLLM

**No measurements.**

Expected server (when runnable): CUDA `vllm serve` on the **same** 5090,
same `Qwen/Qwen3-4B` snapshot, `--dtype bfloat16`,
`--no-enable-prefix-caching`, `--max-model-len 16384` (contract §5.3).

Expected client: **identical** argv as §3, `TAG=vllm`, `PORT=8102`.

| L | n | warmup | TTFT mean/med/p50/p90/p95/p99/std/min/max | TPOT … | ITL … | E2EL … | out tok/s |
|---|---|---|---|---|---|---|---|
| 1024 | — | — | **NOT RUN** | | | | |
| 4096 | — | — | **NOT RUN** | | | | |
| 8192 | — | — | **NOT RUN** | | | | |

Record at first successful run: `vllm.__version__`, attention backend log
line, whether cudagraph capture ran, prefix-cache hit rate (must be ~0).

Raw JSON: none.

---

## 5. Fairness at this session

Nothing to compare. Structural mismatches that will apply on first run
(already in the contract):

- plow Qwen chat template = Gemma-4 markers; vLLM = ChatML.
- sm_120 Qwen prompt consumption = decode-loop (or trap if `_pf` loaded);
  vLLM = real prefill.

Those must appear next to any future ratio.

---

## 6. Reproducibility

Not demonstrated. No JSON artifacts, no second trial, no stddev from this
harness on this GPU.

To produce a reproducible baseline, run the contract twice on an exclusive
5090 and require cell medians within a stated band (script history:
without warmup, identical vLLM config moved mean TTFT **3.3×**).

---

## Appendix A — PRIOR CAMPAIGN, NOT THIS CONTRACT

Source: `perf-data/plow-vs-vllm-baseline.md`, `perf-data/vllm-docker-baseline.md`.
Hardware: **MI350X / gfx950**. vLLM: docker **0.11.2** (`rocm/vllm:latest`),
`vllm bench serve`, conc=1, out=128, **3 prompts/point**, mixed
TTFT-as-prefill labeling. plow “campaign” prefill = **pure prefill**, not
HTTP TTFT. The writeup itself flags a **0.11.2 docker vs 0.25.1 in-process**
discrepancy.

**Do not use as Agent 3–6 baseline.**

### A.1 vLLM docker TTFT / TPOT (Qwen3-4B, gfx950, 2026-07-15)

From `perf-data/vllm-docker-baseline.md` (their labels: TTFT = “prefill”,
TPOT = decode/tok). **3 prompts/point, no warmup documented.**

| ctx | TTFT ms (mean) | TPOT ms |
|---|---:|---:|
| 1024 | 20.63 | 3.150 |
| 4096 | 50.51 | 3.260 |
| 8192 | 87.63 | 3.390 |

No median / p90 / p95 / p99 / std / min / max in that table.

### A.2 plow campaign vs that vLLM column (same doc family)

From `perf-data/plow-vs-vllm-baseline.md`. plow column = campaign prefill
ms (not HTTP). Ratio = campaign / vLLM docker TTFT.

| ctx | plow main prefill ms | plow campaign prefill ms | vLLM docker TTFT ms |
|---|---:|---:|---:|
| 4096 | 222 | 148 | 51 |
| 8192 | 651 | 356 | 88 |

Decode (ms/tok) at 4k: plow campaign 4.7 vs vLLM TPOT 3.26.

These are **AMD**, **different instruments**, **n=3**. They are not RTX 5090
HTTP results.

---

## Appendix B — Inspection commands that did / did not run

Ran (or file-read substitutes): `ls` of `/dev/nvidia*`, sysfs GPU names,
`flake.lock`, CUDA header, driver version, conda torch metadata.

Blocked by Auto-review: `uname`, `nvidia-smi`, `nix --version`,
`nix develop`, `python3 --version`, `env`, `find`, `docker`.

No benchmark process was started. No weights were downloaded. No inference
code was modified.
