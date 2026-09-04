# Prefill campaign summary — Gemma-4 (RTX 5090 / H100), Kimi-K3 & GLM-5.3-Flash (MI355X)

Consolidates 16 individual campaign reports (2026-08-25 to 2026-09-04) from the `shaswot/prefill`
and `shaswot/glm-5.3` branches into one reference. Each subsection cites the original report(s) it
replaces; those files have been removed from `perf-data/` — recover full blow-by-blow detail
(exact commands, register/spill dumps, every rejected knob) via `git log` on this path if needed.

## 1. Gemma-4-12B / RTX 5090 (sm_120a)

*Replaces: `gemma4-12b-sandbox-5090-2026-08-25.md`, `prefill-occupancy-handoff-2026-08-25.md`,
`prefill-vllm-iteration-2026-08-25.md`, `prefill-single-setting-win-2026-08-25.md`,
`prefill-beats-vllm-w8a8-2026-08-25.md`, `prefill-bf16-gap-attribution-2026-08-26.md`,
`prefill-kernel-sweep-2026-08-26.md`, `sm120-beat-vllm-baseline.md`,
`sm120-bf16-gap-findings-2026-08-26.md`, `sm120-iter1-fastpf-routing-2026-08-26.md`,
`sm120-iter2-ws-gemm-rejected-2026-08-26.md`, `sm120-iter8-object-tuner-schema.md`,
`sm120-prefill-w8a8-multictx-2026-08-26.md`.*

**Box**: RTX 5090, sm_120a, 170 SMs, 32 GB GDDR7, driver 580.65.06, CUDA 13.0. `google/gemma-4-12B-it`
(dense, 48 layers: 40 sliding hd256/kvh8/window=1024, 8 full hd512/kvh1, `attention_k_eq_v=true`).
vLLM 0.27.0.

### Real bugs found and fixed
- **`PLOW_UNISEG=1` was required but not defaulted** for `--hf-dir --arch sm_120a`: a plain
  README-quickstart compile produced an asset that failed at *serve* time (different binary) with
  a message not naming the missing flag. **Fixed** in `crates/plowc/src/main.rs`: defaults
  `PLOW_UNISEG=1` whenever `--arch` starts with `sm_120` and the env var isn't already set.
- **vLLM 0.27.0 had 3 bugs blocking Gemma-4-12B entirely** in this environment (transformers
  heterogeneous-config strictness; flashinfer Python 3.11 incompatibility; full-attention layers'
  `q_norm`/`k_norm` allocated at the wrong head_dim — a real version-skew bug between vLLM's
  `gemma4.py` and transformers 5.15.1's new per-layer config schema, worth reporting upstream).
  All three worked around/fixed via an in-process launcher patch, no edits to installed packages
  (`perf-data/tools/vllm_gemma4_launch.py`).

### Live same-box vLLM comparison (first pass, untuned)
vLLM won everywhere: decode 33-37% faster (c1-c16), prefill 28-39% faster (2k/8k/16k ctx).
`GV_MM_MAX=16` (decode GEMV weight-tile-residency, register-neutral) recovered +15-29% decode
throughput but barely moved the vLLM ratio (still HBM-bandwidth-bound). Contradicted an older
`px15` finding on a different physical 5090 — flagged, not re-resolved.

### Prefill: from 31% behind to 27% AHEAD (one fixed setting: input-len 8192, concurrency 1)
| config | TTFT (ms) | vs vLLM (~1204-1221ms) |
|---|---|---|
| original chunked baseline | 1744.3 | 31% behind |
| + single-chunk bucket (`PLOW_MAX_CHUNK=8192`, eliminates per-chunk weight restream) | ~1474 | 18% behind |
| + `PGM_BN=192` (wider N-tile, bandwidth win, real reg/spill cost) | ~1437 | 19% behind |
| **+ true w8a8 (fp8 weights+activations, native `mma.m16n8k32.e4m3`)** | **~959** | **plow 27% FASTER** |

w8a8 GSM8K-gated at N=200: 96.0% vs bf16's 96.5% — within noise, not a real accuracy cost.
w8a16 (fp8 weight, bf16 activation — dequant tax, no native-instruction speedup) was tried and
correctly rejected as a regression (+7-8% slower). **Win extended across context lengths**
(2048/8192/16000): 1.07x / 1.27x / 1.29x faster than vLLM, margin growing with context. Does
**not** close the separate decode gap (vLLM still ~33-37% faster on decode/throughput at c8/c16) —
that needs independent GEMV-kernel work, not attempted.

### Why bf16-vs-bf16 (matched precision) still trails ~18-19%
Isolated microbenches (no `ncu` — permanently `ERR_NVGPUCTRPERM`-blocked in every sandbox this
campaign touched), full-grid, real shapes:
- **GEMM at 66-71% of cuBLASLt** on identical shapes (gate|up 71.0%, down 66.7%, q_full 67.0%,
  o_full 65.7%) — same magnitude gap the w8a8 body shows against cuBLASLt separately.
- **Flash-attention at only 34-44% of the raw bf16 mma hardware ceiling** (hd256 sliding 34.1%,
  hd512 full 43.9%) — a bigger relative gap, plausible given its harder softmax/P·V dependency
  chain.
- Neither is a hardware ceiling: sm_120a has no TMEM/`tcgen05`/`wgmma`, so cuBLAS on this GPU is
  also restricted to `mma.sync`-class kernels — the gap is cuBLAS's autotuned kernel-selection
  maturity vs plow's fixed hand-picked tile, an engineering-depth gap.
- Cheap knobs exhausted: `PLOW_NV_FA256_BKV=64` fails a hard `static_assert` (dead on sm_120a's
  softmax reduction); `PGM_STAGES=4`/`PGM_GLU_STAGES=3` fail the smem load gate (no headroom left
  at the committed tile width); `PGM_BN_GLU=64` (won on the w8a8 body in other reports) measured
  **+10.5-10.8% WORSE** on bf16 — the lever's sign flips with precision (bf16 is more
  bandwidth-bound, not register-bound, at this tile).
- **Real, still-open engineering bug**: a warp-specialized producer/consumer GEMM
  (`px22_ws_stage_bench.cu`'s technique, 1.144x isolated, bit-exact, 0 spills) is fast and correct
  standalone but **hangs specifically when dispatched from the real interpreter megakernel** — 7+
  standalone hypotheses (shape, grid width, cold DRAM, repeated calls, cooperative launch, cp.async
  drain, interleaved call patterns) all ruled out; the trap is believed to be in the interpreter's
  cross-block gate/counter dispatch machinery, unreachable by any standalone probe tried.
  `cuda-gdb`'s CUDA backend and `ncu` both unavailable — root cause needs a minimal synthetic
  multi-op harness (not built) or profiling access neither sandbox in this campaign had. Any future
  `mbarrier`-based technique should be treated as carrying this same undiagnosed risk.
- Segmented-GEMM occupancy-2 (`PLOW_NV_SEG_GEMM`) was evaluated as an alternative and rejected: FLOP-weighted end-to-end payoff is only ~1.05x (~1.5% of total prefill time) against real unbuilt infrastructure cost (no sm_120a `_pfseg`/`_pfgemm` cubin path, no `2×n_cu` packet re-slicing) — not worth it.
- Full kernel-body sweep confirmed only GEMM and flash-attention matter (every other op is
  sub-1%-of-wall-clock, already documented elsewhere). Verdict: port cuBLAS/FlashAttention-2's
  *techniques* (TMA staging, software pipelining) into plow's own inlined kernel bodies — not
  external library linkage (a prior segmented-launch experiment lost ~60% of one op-class's time to
  per-launch floor overhead). Real, multi-session, correctness-risky kernel work — not started,
  gated on explicit go-ahead.

### Infrastructure added, still standing
- `crates/tunedb/src/object.rs` — typed `ObjectCell`/`ObjectConfig`/`ObjectMeasurement`/
  `ObjectRanking` schema (ranks strictly on `end_to_end`, never `isolated`/`complete_object`),
  gated through the same correctness/sample-count publish check as other `tunedb` record kinds.
  No live sweep data yet (`tuning/README.md` notes the old `prefill_tile_measurement.jsonl`'s 13
  rows are not migrated — they carry `"correctness":"unchecked"` and would fail the new gate
  honestly rather than pass by format conversion).
- `scripts/regcheck_sm120.sh` — `ptxas -v` register/spill/stack wrapper for CUDA (CUDA-side
  analog of the existing AMD `regcheck_prefill.sh`).
- `runtime/bench/nvidia/bf16_gemm_vs_cublas_bench.cu`, `fa_prefill_bench.cu` — the standalone
  microbenches behind the GEMM/flash-attention headroom numbers above (not yet wired into
  `CMakeLists.txt`).
- Confirmed vLLM's actual attention backend on this box is `TRITON_ATTN` (FA4 refuses Gemma-4's
  heterogeneous head_dim), not FlashAttention — a materially easier bar than FA3/4 would have been.
- Iteration 1 found `PLOW_FP8_KV_FASTPF=ON` is safe on all-layer fp8-KV packets (live-verified,
  short + long/ring-wrap prompts) — the trap it guarded against was already fixed by a prior PX-23
  change; only stale docs/comments described it as still trapping. Fixed the docs, no code change.

## 2. Gemma-4-31B / H100 (sm_90a)

*Replaces: `gemma31b-h100-status.md`, `gemma4-31b-h100-prefill-baseline-2026-09-04.md`.*

**Box**: H100 80GB HBM3, sm_90a, 132 SM, driver 595.91.07, CUDA 13.2. `google/gemma-4-31B-it`
(dense, 60 layers: 50 sliding window=1024, 10 full), vLLM 0.28.0. First-ever plow build for this
model+arch.

### Baseline (bf16 vs bf16, concurrency 1, no tuning)
| input | plow TTFT (ms) | vLLM TTFT (ms) | ratio | plow TPOT (ms) | vLLM TPOT (ms) | ratio |
|---|---|---|---|---|---|---|
| 2048  | 942.54  | 211.73  | 4.45x slower | 32.85 | 23.77 | 1.38x slower |
| 8192  | 4005.20 | 814.25  | 4.92x slower | 33.50 | 23.18 | 1.45x slower |
| 16000 | 8882.37 | 1855.62 | 4.79x slower | 34.21 | 22.28 | 1.54x slower |

Root cause: `plowc` emits prefill buckets `[128,512,1024]` for this model (derived from its
`sliding_window=1024`, not `--max-ctx`) — every test prompt served as many small chunks with
per-chunk weight re-streaming, the worst-case pattern from the 12B campaign.

### Tuning campaign — 11 stages, TTFT 4.45-4.92x → 1.44-1.55x behind; TPOT ~unchanged
| stage | lever | result |
|---|---|---|
| 1 | `PLOW_MAX_CHUNK=8192` (single-chunk bucket) | TTFT only 10-18% faster — chunk overhead is NOT the dominant cost here (unlike the 12B campaign) |
| 2 | Decode tuner sweep (`GF_FULL` pairing bug found+ruled-out as a red herring, `GV_UNROLL`/`NS_FULL_ABS` sweep) | no net decode improvement — already at local optimum |
| 3 | Segmented TMA-GEMM (canonical GH200 recipe) | **~4.7-5x regression** — reverted immediately |
| 5 | `PLOW_PF_GFUSE=1` (sandwich-norm fusion, already-decode-proven op, prefill opt-in never set before) | small real win, -0.4 to -0.7% TTFT |
| 6 | Bisected Stage 3's regression: FATLITE was register-starving the fat `_pfseg` object; dropping it + actually enabling the separate packet-side `PLOW_TMA_GEMM=1` flag (object-flag alone is inert) | **TTFT 15539→2906ms @8192, then →1999ms with TMA+gfuse — from ~4.0x to ~2.2-2.5x behind**, biggest single win. TPOT regressed 1.4-1.5x→1.55-1.72x (unexplained) |
| 7 | Root-caused Stage 3's original WS384 compile failure: a `#if PLOW_NV_W8A8` guard in `op_gemm_sm90.cuh` (line 583-1680) wraps the bf16 WS384 bodies too — dropping `PLOW_BUILD_W8A8` (the docs' own "bf16-only" guidance) silently deleted them. Fix: build WITH `PLOW_BUILD_W8A8=1` (compiles both precision arms) but never set `PLOW_W8A8=1` at emit (packets stay 100% bf16) | **TTFT →1257-2703ms — 1.46-1.54x behind, another huge win** |
| 8 | `PLOW_SEG_CLASS_SLICE` (occ-2 re-slicing) confirmed dead weight on WS384 (which runs occ-1 anyway) and dropped | **TPOT regression fully recovered**, TTFT unchanged — best config on every metric |
| 9 | Chunk resweep (4096, no better); `PGM90_UNI256_NS=3` (ring depth, default 4) | small ctx-scaling win, adopted; `NS=5` fails to load (smem over device cap) |
| 10-11 | Structural search for a decode-side hidden-guard bug (same pattern as the 3 prefill bugs found) | **none found** — decode's `op_gemm.cuh` has no equivalent guard-trap; `PLOW_NV_FA_KUN=2` confirmed dead code (SASS-identical); `PLOW_NV_FA_GF_FULL=8` structurally infeasible for this model's GQA=2 sliding layers. Decode's ~1.4-1.55x gap is judged a genuine bandwidth-efficiency floor (plow ~56-58% of H100 HBM3 vs vLLM's ~80-85%), not a quick find-and-flip |

**Final config**: cubins `assets-run/gemma4-31b-seg-ws384-ns3/`, packet
`assets-run/gemma4-31b-seg-ws384-ns3-asset/` (`PLOW_MAX_CHUNK=8192 PLOW_TMA_GEMM=1 PLOW_PF_GFUSE=1
PLOW_NO_GLU_FUSE=1 PLOW_SEG_PURE_GEMM=1 PLOW_SEG_FA512=all`, no `PLOW_UNISEG`, no
`PLOW_SEG_CLASS_SLICE`). Final numbers:

| input | plow TTFT | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | 327.00  | 211.73  | 1.545x slower | 32.97 | 23.77 | 1.387x slower |
| 8192  | 1243.86 | 814.25  | 1.528x slower | 33.52 | 23.18 | 1.446x slower |
| 16000 | 2668.36 | 1855.62 | 1.438x slower | 34.26 | 22.28 | 1.538x slower |

Three real, root-caused bugs fixed this campaign (FATLITE register starvation, TMA's missing
packet-side flag, the `W8A8` guard trapping bf16 WS384 bodies). Decode investigated exhaustively;
remaining gap needs either `ncu`/`cuda-gdb` (unavailable every session) or the scoped
`PLOW_NV_LEAN_DECODE` segmented-decode port (real new engineering — MoE-specific machinery, best
precedent on a different, MoE model only reached ~1.32x behind, not parity — not attempted).

## 3. Kimi-K3 & GLM-5.3-Flash / 8x MI355X (gfx950), TP8

*Replaces: `vllm-k3-glm53-baseline.md`.*

Measured 2026-08-31/09-01, `vllm bench serve`, coherence-gated, concurrency 1, `--random-output-len
128`, contexts 8k/16k/64k/128k. Raw CSVs kept at
`perf-data/vllm-rocm/_home_shaswot_models_{Kimi-K3,GLM-5.3-Flash}_bf16_tp8_ctxsweep_c1.csv`.

**Kimi-K3** (vLLM `rocm/vllm:...vllm_0.27.0`, natively registered, no AITER):
| ctx | vLLM TTFT (ms) | vLLM prefill tok/s | vLLM TPOT (ms) | vLLM decode tok/s | plow TTFT (ms) | plow prefill tok/s | plow TPOT (ms) | plow decode tok/s |
|---|---|---|---|---|---|---|---|---|
| 8k   | 994.4    | 8238.5 | 250.59 | 3.99  | 4691.5   | 1746.1 | 58.53 | 17.09 |
| 16k  | 2034.4   | 8053.3 | 250.74 | 3.99  | 9342.7   | 1753.7 | 58.87 | 16.99 |
| 64k  | 8875.3   | 7384.1 | 251.54 | 3.98  | 44739.9  | 1464.8 | 60.82 | 16.44 |
| 128k | 19943.2  | 6572.3 | 252.17 | 3.97  | 111680.5 | 1173.6 | 62.89 | 15.90 |

- **Decode**: plow wins big (16-17 tok/s vs vLLM's ~4 tok/s) — but vLLM's number is a bring-up
  floor, not a ceiling: it's running K3's MXFP4 MoE through a software emulation backend on ROCm,
  not a native kernel.
- **Prefill**: plow loses, 4.4-6.2x — new information, consistent with plow lacking the KDA
  prefill scan vLLM has (69 of K3's 93 layers are KDA).
- plow build caveats: tuning DB stale (fell back to the analytical tile model, not a measured
  optimum), no Lean correctness verification run, a real object-selection gap (`AttnRes` op
  dispatches through the 8-wave interpreter fallback instead of the intended K3 flash object —
  `PLOW_HSACO_K3=ON` doesn't cover this specific flash/GQ object), and a rejected-request trap at
  128k (blob emitted at exactly `--max-ctx 131072`, needed `+2048` margin) — caught via the
  `gen_toks == num_prompts × output_len` gate and fixed by re-emitting.

**GLM-5.3-Flash** (no upstream vLLM support as of 2026-08-31; generic ROCm image's Transformers
bridge crashes on KDA conv1d weight names — required the vendor per-model image
`vllm/vllm-openai-rocm:glm53-flash`, and needed `--privileged` — without it the process was
externally SIGTERM'd 60-90s in, 5/5 failures, root cause not identified):
| ctx | TTFT (ms) | prefill tok/s | TPOT (ms) | decode tok/s |
|---|---|---|---|---|
| 8k   | 1036.96 | 7900.0  | 14.210 | 70.37 |
| 16k  | 1091.89 | 15005.2 | 14.250 | 70.18 |
| 64k  | 2259.13 | 29009.4 | 14.070 | 71.07 |
| 128k | 2990.15 | 43834.6 | 13.940 | 71.74 |

Prefill tok/s *rises* with context (DSA-style sparse attention decoupling compute from context
length, same signature as GLM-5.2's DSA) — the opposite curve from Kimi-K3's. This is a vLLM-only
baseline (no plow-side GLM-5.3-Flash number in this pass); see `perf-data/gh200-branch-merge-review.md`-adjacent GLM-5.3-Flash bring-up work on the `shaswot/glm-5.3` branch for plow's own numbers.
