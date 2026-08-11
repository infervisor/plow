# C-1 — Decode weight SplitZip: measured KILL at the real decode geometry

Date 2026-07-22. Branch `c1-decode-splitzip`. Model gemma-4-12B, RTX PRO 6000
Blackwell (188 SM, 1535 GB/s achievable). Per the design notes
§C-1. **Directive: decode-only (prefill accepted as a non-win).**

## TL;DR

The codec is **lossless and bit-exact** (proven 3 ways). But fused SplitZip decode
is a **measured performance regression at the geometry the decode megakernel actually
runs** (1 block/SM). It is only faster under heavy grid oversubscription (≥5 blocks/SM)
— a configuration the single persistent WS-GEMV megakernel structurally cannot reach.
**C-1 is KILLED by the plan's S2 criterion** (`<1.15× effective on any major shape` —
measured 0.50–0.88×, i.e. −12 to −50%). No serving VU sweep was run: the plan forbids
perf work past a failed gate, and a 128-VU serving campaign of a kernel that is 23–48%
slower per-op would spend shared-GPU leases to confirm a known regression.

## Gate results (all mandatory gates ran)

| gate | result |
|---|---|
| **C-0 compressibility** (audit) | **GO** — 1.3318× bytes-weighted, worst escape 0.018% |
| **S1 host lossless oracle** (byte-identity round-trip of the SzBlob layout) | **PASS** — 329/329 decode-path tensors byte-identical; negative control (corrupt 1 lo byte) DETECTED |
| **S2 device bit-exact** (sz GEMV output vs bf16 GEMV, real 12B shapes) | **PASS** — 0 mismatching outputs at every shape × MM ∈ {1,2,4,8,16} |
| **S2 device performance** (ratio realization at real geometry) | **FAIL / KILL** — 0.50–0.88× (a regression), see below |
| **ptxas register cliff** (the stated MM16/255 risk) | **PASS** — see table |

### ptxas — gemma decode megakernel, sz kernels wired in (PLOW_NV_SZ=1)

| build | base regs | +SZ regs | spill | stack |
|---|---|---|---|---|
| GV_MM_MAX=8 (shipping default) | 212 | **210** | 0 | 192 B |
| GV_MM_MAX=16 | 234 | **242** | 0 | 192 B |

Under the 255 cliff, 0 spill (the 192 B stack is the escape-list locals, not spill).
The register risk the task flagged is real-but-clear — but moot given the perf kill.

## The decisive measurement — real `gemv_rows_sz` vs `gemv_rows`, 12B shapes

Faithful A/B: the SAME kernels the interpreter instantiates (`op_gemm.cuh`
`gemv_rows<MM>` / `gemv_rows_sz<MM>`), launched with the interpreter's own geometry
(`slice=blockIdx, nblk=gridDim`), real 12B weight bytes, EXP_BASE=109. Full data:
`perf-data/c1-decode-splitzip-kernel-ab.txt`. logical GB/s = uncompressed weight bytes
delivered ÷ time; **speedup = bf16 time ÷ sz time**; real% = speedup/1.331.

### GRID=188 — the REAL decode geometry (1 block/SM; 212-reg megakernel occupancy)

| shape | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| qkv    (K3840)  | 0.76× | 0.68× | 0.53× | 0.50× | 0.87× |
| o_proj (K4096)  | 0.76× | 0.68× | 0.60× | 0.54× | 0.88× |
| gate/up (K3840) | 0.77× | 0.68× | 0.59× | 0.51× | 0.87× |
| down   (K15360) | 0.62× | 0.51× | 0.70× | 0.77× | 0.85× |

**sz is slower at every shape and every batch rung.** bf16 GEMV already runs at
~1425–1484 GB/s at B=1 = **~96% of the 1535 GB/s ceiling** — there is no load shadow
left to hide the reconstruct. The sz path lands compute/issue-bound: at B=1 it delivers
~910–1135 logical GB/s (≈ 685–850 GB/s of *actual* compressed bytes), i.e. it reads
fewer bytes but at lower throughput because the ~6-ALU-ops/element reconstruct
(`sz_expand8` shifts/compose + per-chunk escape scan) exceeds the plan's own budget of
"**<~2 spare SASS ops/element**" (Appendix §2/§5). Escapes are lossless-correct but add
a per-8-elem scan over the row's escape slice.

### GRID=1020 — oversubscribed (the setting the in-tree `splitzip_gemv.cu` 1.30× used)

| shape | B=1 | B=8 | B=16 |
|---|---|---|---|
| qkv    | 1.08× | 0.96× | **1.27×** |
| o_proj | 1.12× | 1.10× | **1.28×** |
| gate/up| 1.09× | 1.01× | **1.28×** |
| down   | 1.10× | 0.94× | **1.26×** |

**With ~5.4 blocks/SM of oversubscription sz becomes faster (up to 1.28× at B=16,
95–96% ratio realization)** — reproducing the in-tree positive result. The extra
resident warps supply the memory-level parallelism that overlaps the reconstruct ALU
with the (now-unsaturated) weight loads.

## Why this kills C-1 for the current architecture

The plow decode path is **one persistent WS-GEMV megakernel at 212 registers → 1
resident block/SM** (256 thr × 212 reg ≈ 54 K of 64 K regs/SM). That is the geometry in
the GRID=188 table, and there sz loses. The win only appears at ≥~5 blocks/SM, which a
1-block/SM megakernel cannot provide without abandoning the fused-interpreter design
(and its own scheduling/fusion wins). So the in-tree "**1.300× fused bit-exact**"
headline was a **grid-oversubscription artifact**, not a property available to the
decode engine as built.

This is consistent with — and extends — the plan's own evidence: T6 prefill
w8a16-dequant-in-smem was +0.4..3.5% slower; DFloat11/Cloudflare-Unweight fused decode
was 2×/−40% slower at batch 1; the plan's instruction-budget note predicted <2 ops/elem
available vs ~4 needed. The new datum is that **fixed-length splitzip at the real decode
occupancy also loses, and the crossover is grid-occupancy, not batch size.**

## Honest negatives / what was NOT done and why

- **No serving VU {1..128} sweep.** S2 is a hard kill (a per-op regression); the plan
  forbids perf claims past a failed gate. A serving campaign would need the (large)
  emitter + on-load host-encode loader integration and multiple shared-GPU leases only
  to confirm a slower kernel end-to-end. The effective-BW conversion is already answered
  at the kernel level: sz reads 1.33× fewer bytes but at lower BW, netting a slowdown.
- **fp8 twin path (C-3fp8) is separately NO-GO** from C-0 (ratio 1.042× < 1.12).
- **KV (C-2) not audited** (needs a GPU-side KV dump); independent of this result.

## Reopen criteria (for a future engineer)

C-1 would be worth revisiting ONLY if the decode GEMV moved off the 1-block/SM
persistent megakernel to **oversubscribed standalone GEMV launches** (≥5 blocks/SM),
where the reconstruct hides and the 1.33× converts (up to +28% at B=16 measured). That
is a decode-architecture change, not a codec change, and it trades away the megakernel's
fusion/scheduling advantages — out of scope here.

## Artifacts (all flag-gated OFF by default; bf16 build byte-identical)

- `runtime/nvidia/op_gemm.cuh` — `gemv_rows_sz` / `gemv_glu_rows_sz` / `sz_expand8` /
  `sz_escape8` / `sz_blob` (self-describing blob: header{nesc,exp_base} + lo|cd|eoff|
  epos|eval), `d_gemv_sz` / `d_gemv_glu_sz`.
- `runtime/nvidia/interp_sm120.cu` — `PLOW_DOP_GEMV_SZ` / `_GLU_SZ` arms under `#if
  PLOW_NV_SZ` (compiled out by default).
- `runtime/common/dev_isa.h`, `crates/packet/src/dev.rs` — opcodes 78/79.
- `runtime/tests/sz_batch_sm120.cu` — the faithful A/B (the evidence above).
- `runtime/nvidia/experiments/splitzip_gemv12.cu` — 12B-shape microbench refresh.
- `perf-data/tools/{compress_audit,sz_oracle}.py`, `perf-data/c0-compress-audit.*`.
