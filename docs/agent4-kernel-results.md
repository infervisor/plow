# Agent 4 — kernel results (NVIDIA Qwen prefill)

Contract: `docs/agent2-benchmark-contract.md` (immutable). Profile:
`docs/agent3-profile.md`. Prior A/B: `docs/baseline-results.md`.

Date: **2026-08-18**. Git: `bc39624` + dirty worktree (this agent).

**No HTTP TTFT/ITL/E2EL cell was measured. This file does not claim a win
versus vLLM.** Canonical `vllm bench serve` was **not run** (blockers in §5).

---

## 0. Optimization record

### Optimization 1 — instantiate Qwen hd=128 FLASH_PREFILL in the sm_120 `_pf` object

| field | value |
|---|---|
| **Optimization** | Dispatch `d_flash_prefill_mux<128,64,32>` (plus Gemma-object `HEADNORM_ROPE` hd=128 non-interleaved and `FLASH_MERGE<128>`) so a Qwen3-4B prefill bucket does not `__trap()` in `interp_sm120_pf`. |
| **Hypothesis** | Agent 3 P0: NVIDIA Qwen TTFT is decode-loop prefill because default `_pf` traps on `i[6]==128`. Instantiating the existing mma.sync flash-prefill body at hd=128 unblocks GEMM+FLASH prefill. |
| **Source files** | `runtime/nvidia/interp_sm120.cu` (dispatch). Tests: `runtime/tests/sm120_interp_op_test.cu`. Body already in `runtime/nvidia/op_attention.cuh`. |
| **Before (HTTP)** | **NOT RUN.** Agent 3 DERIVED decode-loop prefill ~6.71 s / 27.4 s / 56.3 s at L=1024/4096/8192. Contract path if `_pf` present: trap. If `_pf` absent: O(n) T=1 launches. |
| **After (HTTP)** | **NOT RUN.** Cubin + op-level oracle only. |
| **Delta** | **No served TTFT/ITL/E2EL delta.** Cannot be computed without the Agent 2 client. |
| **Correctness** | hd=128 flash-prefill oracle **PASS** (relL2 ~1.7e-3, tol 2e-2). `_pf` cubin built; symbol `_Z15interp_sm120_pf11PlowProgram` present. Full `sm120_interp_op_test` TU **did not compile** (pre-existing MoE wrappers). |
| **Decision** | **KEEP** the dispatch change (correctness + cubin). **Do not promote as a vLLM beat.** Re-measure with Agent 2 argv after Qwen3-4B assets + CUDA `plowrt` exist. |

---

## 1. Hardware / environment (this session)

| item | value | how |
|---|---|---|
| GPU | 1× NVIDIA GeForce RTX 5090, 32607 MiB, PCI `0000:41:00.0`, UUID `GPU-4ac5dc0e-…9b3a` | `nvidia-smi` |
| Visible device | index 0 (`/dev/nvidia2`) | `nvidia-smi -L` |
| Driver | 580.142 Open Kernel Module | `/proc/driver/nvidia/version` |
| Host CUDA | 13.0, `nvcc` V13.0.48 (`/usr/local/cuda`) | `nvcc --version` |
| Nix CUDA (cubin) | 12.9, V12.9.86 (`PLOW_NVCC` from `nix develop`) | `$PLOW_NVCC --version` |
| Nix | 2.35.2 | `nix --version` |
| nixpkgs | `49a4bd0573c376468dd7996ddb6f9fa31d8c4d97` | `nix flake metadata` |
| cargo (dev shell) | 1.95.0 | `nix develop` hook |
| OS | Ubuntu 22.04.5, Linux 6.8.0-65-generic | `/etc/os-release` |
| Persistence | Enabled | `nvidia-smi` |
| Clocks at idle | 225 / 405 MHz, P8, 6 W | before work |
| Clocks after oracle | 2512 / 14001 MHz, P0, 32 W | after hd128 test |
| App clocks | **not locked** | contract note |
| `CUDA_VISIBLE_DEVICES` / `PLOW_*` | unset | `env` |
| Qwen3-4B weights | **none** (no `*.safetensors`, no HF cache, no `/root/models`) | find |
| `plowrt` | **missing** (`target/release/plowrt`) | ls |
| CUDA vLLM server | **missing** (`import vllm` fails; nix `.#vllm` is client-only) | python / contract |

Same visible 5090 as Agents 2–3. Other host 5090s (`81:00.0`, `c1:00.0`) are in sysfs; `nvidia-smi -L` showed only GPU 0.

---

## 2. What changed in source

`runtime/nvidia/interp_sm120.cu` (Gemma **prefill** object, `PLOW_NV_PREFILL=1`):

- `PLOW_DOP_FLASH_PREFILL` → `d_flash_prefill_mux<128, 64, PLOW_NV_FA128_BKV>` with `PLOW_NV_FA128_BKV=32`.
- Gemma `HEADNORM_ROPE`: hd=128, `i[5]==0` (Qwen non-interleaved).
- Gemma `FLASH_MERGE`: hd=128.
- Qwen **decode** object (`PLOW_NV_PREFILL=0`) is not in this `#if` for FLASH_PREFILL.

`runtime/tests/sm120_interp_op_test.cu`: added `test_flash_prefill<128,64,32>` / varlen cases (causal tile-exact, ragged, GQA4, chunked, soft `scale=0.0884`, ns=3 merge).

---

## 3. Cubin rebuild (existing process)

Canonical served-object path: `runtime/CMakeLists.txt` `PLOW_SM120_CUBIN=ON` target `sm120_cubins`, driven from `nix develop` (same table as `scripts/build_sm120_cubin.sh`).

```bash
nix develop --command bash -lc '
  cmake -S runtime -B build-agent4-cubin \
    -DPLOW_SM120_CUBIN=ON \
    -DPLOW_CUBIN_NVCC="$PLOW_NVCC" \
    -DPLOW_CUBIN_DIR="$(pwd)/build-agent4-cubin/cubin"
  cmake --build build-agent4-cubin --target sm120_cubins -j2
'
```

Result (**PASS**, ~94 s):

| artifact | size | kernel gate |
|---|---:|---|
| `build-agent4-cubin/cubin/interp_sm120_pf.cubin` | 814344 B | `_Z15interp_sm120_pf11PlowProgram` present |
| `build-agent4-cubin/cubin/interp_sm120.cubin` | 2764848 B | `_Z12interp_sm12011PlowProgram` present |
| `build-agent4-cubin/cubin/sample_sm120.cubin` | 52432 B | `plow_sample` present |

`_pf` defines: `-DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2 -DPLOW_NV_EMBED_SMEM=1 -DPLOW_NV_PREFILL=1`.

`cuobjdump --dump-resource-usage` on `_pf`:

```
Function _Z15interp_sm120_pf11PlowProgram:
  REG:240 STACK:1024 SHARED:3344
```

Cubin files are gitignored (`*.cubin`, `build-*/`). Rebuild from this worktree to reproduce.

Failed cmake path (not used for cubins): `cmake -DPLOW_CUDA=ON` → `sm120_interp_op_test` died in `cuda_fp16.h` (`_NV_IF__NV_TARGET_BOOL_NV_IS_DEVICE`). Same nix `CPATH` vs CUDA-math clash `nvcc_cubin.sh` / `perf-data/px6-sm-quantization.md` already document. Cubins avoid it via `env -i`.

---

## 4. Numerical / correctness

### 4.1 Canonical TU compile — FAIL (pre-existing)

```bash
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
unset CPATH LIBRARY_PATH LD_LIBRARY_PATH
nvcc -arch=sm_120a -O2 -std=c++17 -I runtime/common -I runtime/nvidia \
  runtime/tests/sm120_interp_op_test.cu -o build-agent4-tests/sm120_interp_op_test
```

Errors (not introduced by hd=128 cases; also recorded in `perf-data/px6-sm-quantization.md`):

- `sm120_interp_op_test.cu:632` `d_moe_router_gemma_score_fast` too few args
- `:813` `d_moe_expert_glu_norm_gemma` too few args
- `:849` `d_moe_expert_down_gemma` too few args
- `:1208` `d_quant_fp8` `const bf16*` vs non-const

**Not fixed** (unrelated MoE/quant wrappers).

### 4.2 hd=128 FLASH_PREFILL oracle — PASS

Same `d_flash_prefill` / `d_flash_prefill_mux` / `d_flash_merge` templates and the same f32 CPU ref / relL2≤2e-2 gate as `sm120_interp_op_test.cu`. Host nvcc 13.0, `unset CPATH`. GPU: RTX 5090 sm_120.

| case | relL2 | maxabs | result |
|---|---:|---:|---|
| hd128 h4 kv2 len128 causal fused | 0.001681 | 0.00282 | PASS |
| hd128 h4 kv2 len200 ragged | 0.001682 | 0.002767 | PASS |
| hd128 h8 kv2 len128 GQA4 | 0.001674 | 0.002972 | PASS |
| hd128 chunk sq100 skv612 qp0=512 | 0.001734 | 0.002605 | PASS |
| hd128 soft len512 causal scale=0.0884 | 0.002046 | 0.002835 | PASS |
| hd128 len200 causal ns=3 + merge | 0.001674 | 0.002904 | PASS |
| varlen hd128 R4 midtile vs serial | 0.001884 | 0.002833 | PASS + BIT-EXACT |

This is **op-level** correctness, not served tokens, not ChatML, not Agent 2 TTFT.

---

## 5. Canonical benchmark (Agent 2) — NOT RUN

Required client (`docs/agent2-benchmark-contract.md` §5.1): `nix develop .#vllm --command vllm bench serve --backend openai-chat` at L∈{1024,4096,8192}, out=128, conc=1, n=32, warmup=4, seed=0, ignore_eos, temp=0.

| cell | status |
|---|---|
| OUR IMPLEMENTATION (plowrt HTTP) | **NOT RUN** |
| VLLM (same client) | **NOT RUN** |
| Paris coherence gate | **NOT RUN** |
| TTFT / TPOT / ITL / E2EL tables | **empty** |

Blockers (all still true):

1. No Qwen3-4B checkpoint.
2. No CUDA `plowrt` (`cargo build --release -p plowrt --features cuda` not run; no weights to serve).
3. No CUDA vLLM **server** (flake `.#vllm` is client-only).
4. Therefore no like-for-like HTTP A/B.

Do **not** fill TTFT from Agent 3’s derived decode-loop seconds, from gfx950 Appendix A, or from the oracle kernel times.

When assets exist, log which prefill path ran (`prefill object loaded` vs decode-only), require CUDA engine (not CPU), and fill cells with the contract argv only.

---

## 6. Comparison vs baseline / vLLM

`docs/baseline-results.md` HTTP tables are still **NOT RUN**. This session adds **no** comparable row.

| claim | allowed? |
|---|---|
| hd=128 flash-prefill body matches f32 oracle under existing relL2 | yes (measured) |
| `_pf` cubin now contains a hd=128 FLASH_PREFILL arm | yes (built; dispatch present) |
| served Qwen TTFT improved | **no measurement** |
| beats vLLM | **no** — contract A/B not executed |

---

## 7. Commands actually run (complete)

```text
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,power.draw,power.limit,pstate,persistence_mode --format=csv
nix flake metadata
nix develop --command bash -lc '… cmake sm120_cubins …'     # §3 PASS
nix develop --command bash -lc 'cmake -DPLOW_CUDA=ON … sm120_interp_op_test'  # FAIL cuda_fp16.h
nvcc -arch=sm_120a … runtime/tests/sm120_interp_op_test.cu   # FAIL MoE wrappers
nvcc -arch=sm_120a … hd128 FLASH_PREFILL oracle               # PASS §4.2
cuobjdump --dump-resource-usage build-agent4-cubin/cubin/interp_sm120_pf.cubin
```

Weights search: `/root/models`, `/workspace/models`, `$HOME/.cache/huggingface`, `/data`, `/mnt` — no `*.safetensors`.

---

## 8. Follow-up (not done)

1. Fix or skip the four MoE/quant wrappers so `sm120_interp_op_test` links as a ctest.
2. Emit Qwen3-4B sm_120 assets with `--max-ctx ≥ 16384`; place `interp_sm120_pf.cubin` next to them.
3. Private CUDA `plowrt`; confirm log is not `CPU reference backend active`.
4. Run Agent 2 §5.1 both sides; fill `docs/baseline-results.md` (or a new results file — do not rewrite the contract).
5. Only then may Agent 6 discuss vLLM ratios.
