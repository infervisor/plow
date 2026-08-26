# Iteration 0 — controlled baseline: beat vLLM on sm_120, Gemma-4-12B

Consolidates the existing multi-session campaign on this branch (`shaswot/prefill`) into one
baseline reference, per the mission's Iteration-0 checklist. **This does not re-derive numbers
that are already measured and cited below** — it cites the source report for each, confirms what
was previously unconfirmed (attention backend, `ncu` availability), and adds one new tool
(`scripts/regcheck_sm120.sh`) that didn't exist yet. See "Deviations" for two corrections to the
mission brief's stated assumptions.

## Environment

- Commit: `a3d57b358391980f7e8e03e4f8a6cc5d018b9e82` (branch `shaswot/prefill`), tree clean at
  session start (`git status --short` empty) before this report + `scripts/regcheck_sm120.sh`.
- GPU: **NVIDIA GeForce RTX 5090**, sm_120a, 170 SMs, 32607 MiB GDDR7. Driver 580.65.06, CUDA
  13.0.48 (`/usr/local/cuda`, system nvcc — outside `nix develop`, this repo's own convention for
  the CUDA toolchain). `nix (Nix) 2.35.2`.
- vLLM: **0.27.0**, pip-installed into a venv, launched via
  `perf-data/tools/vllm_gemma4_launch.py` (patches 3 vLLM/transformers bugs blocking Gemma-4-12B —
  see `perf-data/gemma4-12b-sandbox-5090-2026-08-25.md` §3).
- Model/checkpoint: `google/gemma-4-12B-it`, dense, 48 layers (40 sliding hd256 kvh8 window=1024,
  8 full hd512 kvh1 causal-only), `attention_k_eq_v=true`. On disk:
  `/workspace/models/gemma-4-12B-it` (bf16 source), `-it-fp8` (fp8 weight twins),
  `-it-merged` (the combined dir `--rt-checkpoint` points at).

## Deviations from the mission brief's stated assumptions

1. **GPU**: brief assumes RTX PRO 6000 Blackwell, 188 SMs. The only GPU present is an
   **RTX 5090** (170 SMs, same `sm_120a` arch family). One older doc notes RTX PRO 6000 decode
   numbers "reproduce on sm_120a" but was never itself measured on one. Every number in this
   report and this campaign is RTX 5090 — treated as ground truth for this box, not a stand-in.
2. **Orphaned process**: an unrelated `plowrt serve` (PID 60722, from the prior session's w8a8-win
   benchmark, no active connections, no other users on the box) was holding 20 GiB VRAM at session
   start. Stopped (user-executed `kill -TERM`) before any measurement in this report; confirmed
   clean via `nvidia-smi` (2 MiB used, 0% util) and `gpulease --status` (`gpu0: free`) afterward.

## Build flags actually in effect (read from source, not assumed)

| flag | default (this build) | source |
|---|---|---|
| `PLOW_UNISEG` | forced to `1` for any `sm_120*` `--hf-dir` emit unless `PLOW_UNISEG=0` is set | `crates/plowc/src/main.rs:541-545` |
| `PLOW_FP8_KV_FASTPF` (CMake, decode/prefill objects generally) | `OFF` | `runtime/CMakeLists.txt:35` |
| `PLOW_FP8_KV_FASTPF` (the **served** `plow-interp-sm120a` nix package) | **`ON`** | `flake.nix:292` — deliberately different from the CMake default |
| `PLOW_NV_FA_FP8PV` | `OFF` everywhere, incl. the served nix package | `runtime/CMakeLists.txt:241`; not passed in `flake.nix:289-297` |
| `PLOW_NV_FA_PIPE` | `1` (source `#define`, not a CMake option) | `runtime/nvidia/op_attention.cuh:341` |
| `PLOW_NV_W8A8` | `OFF` by default; the w8a8-win asset builds it `ON` explicitly | `runtime/CMakeLists.txt:20` |

## vLLM's actual attention backend — CONFIRMED, not assumed

**`TRITON_ATTN`**, on every serve log checked this campaign (`serve-phase0-vllm.log`,
`serve-phase0-vllm2.log`, `serve-vllm-final-verify.log`):

```
Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
FA4 not available, forcing TRITON_ATTN backend.
...
Using AttentionBackendEnum.TRITON_ATTN backend.
```

Not FlashAttention, not FlashInfer — vLLM's own FA4 backend selector refuses Gemma-4's
per-layer-varying head dim and falls back to its Triton-JIT'd reference attention kernel. This was
previously known only implicitly (buried in uncommented serve logs); it had not been written up as
a fact anywhere in `perf-data/`. **This matters for the mission's "attention ≥90% of vLLM's actual
backend" gate: the bar is a Triton kernel, not a hand-tuned CUDA/CUTLASS kernel** — a materially
easier target than FlashAttention-3/4 would have been, and consistent with why plow's flash-prefill
already contributes to a net TTFT win at w8a8 despite running at only 34-44% of its own hardware
ceiling (see below).

## Nsight Compute — confirmed still blocked, re-verified this session

`ncu --set full` against a trivial probe kernel: `ERR_NVGPUCTRPERM` (no GPU performance-counter
permission in this container), same failure mode documented in three prior sessions
(`perf-data/px4-flash-streaming.md`, `perf-data/gemma12b-gh200-prefill-campaign.md:277`'s
counter-example was a *different* box). Falling back to this campaign's established method
throughout: standalone `.cu` microbenches timed with CUDA events / `clock64`, differential
ablation, and SASS/symbol census — never a hardware counter claim.

## GEMM: real, quantified headroom vs cuBLASLt (both precisions)

Both benches ran full-grid (170 SMs), real Gemma-4-12B production shapes, L2-cold/warm controlled
(both engines cycled 16 weight replicas — the confound was checked and found to move nothing,
<0.5% either way). cuBLASLt was **not** left at its first heuristic result — this is the
autotuned, best-observed cuBLASLt number cited in `perf-data/px9-gemm-body.md` Result 4.

| precision | shape | plow | cuBLASLt | plow/cuBLASLt | source |
|---|---|---|---|---|---|
| w8a8 (fp8x fp8, `mma.m16n8k32.e4m3`) | gate\|up | 317.3 TFLOP/s | 499.8 (99.2% of 503.8 in-tree peak) | **63%** | `px9-gemm-body.md` |
| w8a8 | down | 341.5 | 481.2 (95.5%) | **71%** | ″ |
| w8a8 | q_full | 327.0 | 497.5 (98.7%) | **66%** | ″ |
| w8a8 | o_full | 328.5 | 479.0 (95.1%) | **69%** | ″ |
| bf16 (`mma.sync.m16n8k16`) | gate\|up (GLU) | 169.3 | 238.3 | **71.0%** | `prefill-bf16-gap-attribution-2026-08-26.md` |
| bf16 | down | 156.6 | 234.7 | **66.7%** | ″ |
| bf16 | q_full | 154.4 | 230.5 | **67.0%** | ″ |
| bf16 | o_full | 154.5 | 235.2 | **65.7%** | ″ |

**Neither precision meets the mission's 90%-of-cuBLASLt gate today.** Root-caused (by elimination,
not a counter) to the `cp.async`/`LDGSTS` global→smem operand-staging path — mma throughput,
fragment layout, swizzle, barriers, epilogue and accumulator were independently ruled out.
sm_120a (consumer Blackwell) has no TMEM/`tcgen05`/`wgmma` — cuBLASLt on *this* silicon is also
restricted to `mma.sync`-class kernels, so the gap is a tuning/pipelining-maturity gap, not a
hardware-class mismatch.

## Attention: real headroom vs raw hardware ceiling

Isolated `d_flash_prefill<HD,BQ,BKV>` microbench, real shapes, seq=8192, full-grid, masked-aware
TFLOP/s (source: `perf-data/prefill-kernel-sweep-2026-08-26.md`):

| arm | layers | ms | TFLOP/s | % of bf16 mma ceiling (259.2, PX-9) |
|---|---|---|---|---|
| hd256 sliding (BQ64/BKV32, shipped) | 40/48 | 1.4587 | 88.3 | **34.1%** |
| hd512 full (BQ32/BKV16, shipped) | 8/48 | 9.6732 | 113.7 | **43.9%** |

No live FlashAttention-2 cross-check was possible (no sm_120a-targeted prebuilt wheel, from-source
build not attempted — reported as not attempted, not substituted with a number from another GPU).
This is a *harder-to-reach* ceiling reading than GEMM's (kernel-vs-raw-hardware, not
library-vs-library), not directly ratio-comparable to the GEMM numbers above, but points the same
direction: real, substantial headroom, plausible given flash-attention's harder cross-tile
dependency chain (running softmax stats, P·V using just-computed probabilities).

## End-to-end state today (input-len 8192, concurrency 1, TTFT)

| config | TTFT (ms) | vs vLLM 0.27.0 bf16 |
|---|---|---|
| vLLM 0.27.0, bf16 (TRITON_ATTN, re-verified fresh) | 1220.9 (mean of 3, 1211.8-1233.3) | — |
| plow, bf16, best tuned (single-chunk bucket + `PGM_BN=192`) | ~1437 | 0.85x (19% behind) |
| **plow, w8a8** (fp8 weights + fp8 activations, native `mma.m16n8k32.e4m3`) | **~959 (mean of 10, 943.6-969.4)** | **1.27x — plow 27% FASTER** |

Source: `perf-data/prefill-beats-vllm-w8a8-2026-08-25.md`. GSM8K N=200: 96.0% (w8a8) vs 96.5%
(bf16) — within one standard error, not distinguishable from noise. **The w8a8 win is real,
GSM8K-validated, and already committed (`aca2653`)** — it is a genuine precision change (not
bit-exact vs the bf16 baseline), scoped to this one setting only (not yet retested at other
context lengths/concurrencies — see that report's Open Items).

At *matched* precision (bf16 vs bf16, same setting), plow still trails vLLM by ~18%
(`prefill-bf16-gap-attribution-2026-08-26.md`) — the GEMM/attention headroom tables above are
where that gap lives.

## Cheap knobs already exhausted (do not re-try without new evidence)

- `PLOW_NV_FA256_BKV=64` — fails to compile: `op_attention.cuh:2943`'s
  `static_assert(BKV <= 32, ...)`. Dead on sm_120a's current softmax reduction, not just untested.
- `PGM_STAGES=4`/`PGM_GLU_STAGES=3` (at `PGM_BN=192`) — fails to load:
  `dynamic smem 102400 B exceeds device opt-in limit 101376`. No headroom for deeper pipelining at
  the committed tile width.
- w8a16 (fp8 weight, bf16 activation, dequant-to-bf16-then-bf16-mma) — measured **slower** than
  bf16 (+7-8%): halves weight-read bytes but pays a dequant step and still runs the half-throughput
  bf16 mma. Rejected in favor of true w8a8.
- `PLOW_MOE_PREFILL`/`PLOW_MLA_PREFILL`/family-arm knobs — not applicable (Gemma-4-12B is dense,
  no MoE/MLA arms compiled in for this model).

## New this iteration: `scripts/regcheck_sm120.sh`

No `ptxas -v` register/spill/stack wrapper existed for the CUDA side (only the AMD/HIP equivalent,
`scripts/regcheck_prefill.sh`). Added, following that script's report-only pattern: builds via
`scripts/build_sm120_cubin.sh` with `-Xptxas -v,--warn-on-spills` appended, greps registers/stack
frame/spill stores/spill loads per compiled symbol out of the log.

Verified against the real production w8a8 object (`-DPLOW_NV_W8A8=ON`, matching
`prefill-beats-vllm-w8a8-2026-08-25.md`'s build):

```
object                 symbol                                            regs  stack spillS spillL
interp_sm120.cubin     _Z12interp_sm12011PlowProgram (decode)              255   1024      0      0
interp_sm120_pf.cubin  _Z15interp_sm120_pf11PlowProgram (prefill)          242   1024      0      0
```

Matches the prior report's `REG:240 → 242` (negligible cost from w8a8) and **0 spills confirmed on
the currently-shipped w8a8 prefill object** — satisfies that victory-gate item today, before any
new kernel work starts. Cubin hashes this session
(`build_sm120_cubin.sh <out> -DPLOW_NV_W8A8=ON`, CUDA 13.0.48, commit `a3d57b3`):

```
sha256:b0bd9756e3cbecce0d9e9f7a31e9115ec1488ed91500e722a1ff51698fc524e5  interp_sm120.cubin
sha256:a63ad022f057a861e89f0b83152aa26945d3640f671d5ad4e675262d8c75df32  interp_sm120_pf.cubin
```

Dynamic shared memory was **not** added to this script — it's a host-side launch parameter
(`PLOW_NV_ARENA_FLOATS * sizeof(float)`, `interp_sm120.cu:677-695`), not something `ptxas -v`
reports; the codebase already embeds it in the cubin as a queryable `__device__` global
(`plow_arena_bytes`, read by `exec/gpu.rs:1531` via `cuModuleGetGlobal`) for exactly this purpose.
Deferred until an iteration actually changes smem usage — reuse that existing mechanism then
rather than duplicating it here.

## What's next

The concrete, already-scoped, already-validated-in-isolation candidate is
`runtime/bench/nvidia/px22_ws_stage_bench.cu`'s producer/consumer warp specialization
(1.144x isolated on the plain w8a8 GEMM body, bit-exact, 0 spills) — proven in isolation but
explicitly not yet integrated into `op_gemm.cuh`/`interp_sm120.cu`, and with no end-to-end number.
That's Iteration 2 of `/root/.claude/plans/glimmering-soaring-stream.md`. Iteration 1 (attention
fast-path routing/capability validation) runs first: `docs/flags-reference.md` already documents
that an all-layer fp8-KV packet traps under `PIPE=1` when built with `FASTPF=ON` ("a build cannot
know which packet it will load") — exactly the load-time-vs-trap gap the mission's Iteration 1
asks to close.
