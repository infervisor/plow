# Why plow still trails vLLM in bf16-vs-bf16, and what's left — Gemma-4-12B / RTX 5090 (sm_120a)

Follow-on to `perf-data/prefill-beats-vllm-w8a8-2026-08-25.md`. That report closed the live gap
against vLLM using w8a8 (fp8 weights + activations) — a real, GSM8K-validated win, but a different
precision than vLLM's bf16 default. This report answers the harder question: at the SAME
precision (bf16 vs bf16), plow still trails vLLM by ~18% at the same fixed setting (input-len
8192, concurrency 1: plow ~1437ms vs vLLM ~1221ms). Why, and what's left to close it?

## Research summary (three parallel investigations, this session)

1. **No profiler is available.** `ncu` is still `ERR_NVGPUCTRPERM`-blocked in this sandbox
   (re-verified). The one runtime flag that would give a GEMM-vs-flash-attention wall-clock split
   (`--pf-seg-time`) needs segmented prefill objects that only exist for sm_90a
   (`scripts/build_sm90a_cubin.sh`); `scripts/build_sm120_cubin.sh` has no equivalent — porting it
   is real engineering work, not a flag. `--ttft-log` runs on sm_120a but only reports one
   collapsed `PREFILL` number on the NVIDIA path (`exec/gpu.rs` never calls the finer `PF_*`
   sub-phase instrumentation that exists for AMD).
2. **Why plow isn't structurally behind cuBLAS here**: sm_120a (consumer/workstation Blackwell)
   has no TMEM and doesn't support `tcgen05`/`wgmma` (Hopper/datacenter-Blackwell only) — cuBLAS on
   *this* GPU is also restricted to `mma.sync`-class kernels, same instruction class plow uses.
   FlashAttention-3/4 don't run on sm_120a either. The likely dominant factor is cuBLAS's
   autotuned kernel-selection maturity (tile/split-K/swizzle picked per shape from a huge tuned
   library) vs plow's fixed hand-picked tile — an engineering-depth gap, not a hardware ceiling.
3. **Every remaining `-D`-flag knob is a dead end**, confirmed by code research before spending
   GPU time chasing them: FA512 knobs are Hopper-only dead code on sm_120a; `PGM_BM` widening has
   only ever been measured on the w8a8 body (`perf-data/px13-prefill-gemm.md`) where it regressed
   end-to-end (+19.4% at BM=256, +6.9% at BM=64) despite looking like a win in isolation — the
   same "megakernel resources are global" trap that burned `PGM_BN_GLU=64` last session; segmented
   lean-object paths are structurally incompatible with how this asset actually serves
   (`PLOW_UNISEG=1` mandatory).

## Phase A — isolated bf16 GEMM vs cuBLASLt: real headroom confirmed

Built a standalone microbench (same technique as this session's `PGM_BN_GLU` oracle: no `ncu`
needed), mirroring `perf-data/px9-gemm-body.md`'s method exactly but for bf16xbf16 instead of
w8a8 — full-grid (170 SMs), the real Gemma-4-12B prefill shapes, at the **actual deployed tile
config** (`PGM_BN=192`, `PGM_BN_GLU=128`, this session's winning bf16 setting), M=8192.

| shape | plow (full-grid) | cuBLASLt | plow/cuBLASLt |
|---|---|---|---|
| gate\|up (GLU) | 169.3 TFLOP/s | 238.3 TFLOP/s | 71.0% |
| down | 156.6 TFLOP/s | 234.7 TFLOP/s | 66.7% |
| q_full | 154.4 TFLOP/s | 230.5 TFLOP/s | 67.0% |
| o_full | 154.5 TFLOP/s | 235.2 TFLOP/s | 65.7% |

**plow's bf16 GEMM runs at 66-71% of cuBLASLt's throughput on identical shapes — a real ~30-34%
gap, the same magnitude PX-9 found for the w8a8 body against cuBLASLt.** This directly answers the
question: GEMM has genuine, quantified headroom. It is not the whole ~18% end-to-end gap (prefill
also includes flash-attention, norm/rope, and dispatch, which this isolated bench doesn't cover),
but it is large enough to be a material contributor, consistent in direction and magnitude with
the w8a8 finding that pinned the cause on the `cp.async`/`LDGSTS` operand-staging path rather than
mma throughput, occupancy, or smem bank conflicts (all independently ruled out in that report).

Microbench source kept at
`/tmp/claude-0/.../scratchpad/bf16_gemm_vs_cublas.cu` (session-local scratch, not committed —
promote to `runtime/bench/nvidia/` if this line of investigation continues).

## Phase B — two cheap knob checks, both clean negatives

1. **`PLOW_NV_FA256_BKV=64`** (halve flash-prefill's per-tile barrier/drain count) — **fails to
   compile**: `op_attention.cuh:2943`'s `static_assert(BKV <= 32, "softmax lane-per-kv reduction
   needs BKV <= 32")`. This is a correctness constraint in the current mma.sync-based softmax
   reduction, not a resource limit — the GH200/sm_90a report that measured a win with BKV=64 used
   a completely different kernel body ("n64 score wgmma") that doesn't exist for sm_120a. Dead on
   this architecture, not just untested.
2. **`PGM_STAGES=4`/`PGM_GLU_STAGES=3`** (deeper pipeline, at the current `PGM_BN=192`) — **fails
   to load**: `dynamic smem 102400 B exceeds device opt-in limit 101376` (the devblob load gate
   catching it cleanly, exactly as designed — a loud failure, not a silent wrong answer). Arena
   math confirms why: at `BN=192`/`STAGES=3` the plain-GEMM arena is already ~79.75 KiB of the
   ~99 KiB cap; one more stage adds ~25 KiB more than what's left. No headroom for deeper
   pipelining at the tile width we've already committed to.

Both were single-flag rebuild attempts, ~10-15 minutes total, and both resolved cleanly (compile
error or load-time refusal) rather than requiring a live benchmark to disprove — cheap, as scoped.

## Conclusion — reporting per the approved plan, not building

Per explicit user direction this session, stopping here rather than starting new kernel-body work:

**The ~18% bf16 gap is real, and Phase A shows where a meaningful piece of it lives: plow's bf16
GEMM at ~67% of cuBLASLt's throughput on this exact hardware.** The concrete next lever — same one
already scoped in the prior session's plan and confirmed reachable on this GPU
(`perf-data/px9-gemm-body.md` §Result 7 citing "rtx-01 §1": single-CTA TMA exists on sm_120a even
without `wgmma`) — is a **TMA-based GEMM mainloop port** (`cp.async.bulk.tensor` + mbarrier
producer, `mma.sync` consumer). This would be genuine new kernel-body work with no existing
sm_120a precedent in this tree, and is **dtype-agnostic** — it would help the already-winning
w8a8 path too, not just bf16, since the staging-path bottleneck it targets is independent of
operand precision. Not started this session: real correctness risk (this repo has a documented
history of "fluent but wrong" regressions from similarly-scoped kernel changes), multi-session
scope, needs explicit separate go-ahead before implementation, and would need the full
correctness discipline (numeric oracle + GSM8K at N=200 minimum on the integrated serving path)
before trusting any resulting speed number.

**Worth restating plainly**: w8a8 already beats vLLM (27% faster, GSM8K-validated). If the
practical goal is "serve this model faster than vLLM," that's done. This bf16 investigation
answers a harder, separate question — matching vLLM at *equal* precision — which is valuable to
understand but not required for the win already banked.
