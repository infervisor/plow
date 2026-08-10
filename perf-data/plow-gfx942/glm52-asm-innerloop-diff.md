# ISA-level diff: plow's three hot GLM prefill kernels vs aiter/Tensile asm (gfx942)

> **Scope:** gfx942 objects vs shipped aiter gfx942 asm; no GPU · **CDNA3-SPECIFIC** — an ISA-level diff against gfx942 assembly. Every instruction-mix claim is CDNA3 encoding.

2026-08-08. Instrument: `perf-data/plow-gfx942/probes/asm_loops.py` (backward-branch loop
extraction + instruction-mix classification) over llvm-objdump disassemblies of our
compiled objects (base = worktree-glm52-bringup @ aa26a6c, PLOW_OCC4=1 PLOW_L2HIER=1)
and the shipped AMD asm on this box (`/workspace/aiter/hsa/gfx942/`). The hipBLASLt
Tensile `.co` bundles are compressed (not ELF) — aiter's own hand-written
`bf16gemm_*_bshuffle_splitk` asm stands in as the Tensile-class reference.

Question answered: **are our inner loops optimal at the ISA level, and what do aiter's
asm kernels actually do differently?**

## 1. Flash: `d_flash_mla_prefill_v2` vs `fmha_fwd_hd192x128_bf16_causal` (ts_qo=128, ts_kv=32)

Per inner-loop iteration (aiter loop = 771 insts; ours = the KV-tile body class, 2030-4207
insts, and the tightest QK-chunk subloop, 244 insts):

| metric | plow KV body | plow QK subloop | aiter fmha |
|---|---|---|---|
| MFMA | 64 | 32 | 80 |
| VALU / MFMA | **19–21** | 3.5 | **6.7** |
| gload / MFMA | 0.5 | 0.5 | 0.5 |
| ds_read / MFMA | **2.3** | 0.5 | **0.5** |
| ds_write / iter | 24 | 0 | **4** |
| waitcnt / iter | **128–271, full `lgkmcnt(0..2)`, `vmcnt(0)`×12–74** | 16 | **6, partial `vmcnt(8)`/`vmcnt(12)`** |
| barriers / iter | 4 | 0 | 4 |
| MFMA issue density | 1.5–3% | 13% | 10.4% |

aiter's disciplines: (a) the streamed KV operand flows **global→VGPR** — 40 gloads vs
only 4 ds_writes per iteration (the MoonMath "V from L1" shape, applied to K as well);
(b) softmax VALU is interleaved lean (6.7 total VALU/MFMA *including* softmax vs our
19–21); (c) 8–12 loads stay in flight across the wait points.

## 2. MoE grouped: `d_moe_group_pf_t` k-tile loop vs `fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3`

Ours = 203–232 insts per k-tile; aiter = 3621-inst outer body (whole expert tile).

| metric (per MFMA) | plow | aiter fmoe |
|---|---|---|
| VALU | 7–8 | 5.5 |
| gload | 0.25–0.44 | 0.8 |
| ds_read | 0.75 | 0.33 |
| waitcnt | 0.8–0.9, **all full** (`lgkmcnt(0)`×5 + `vmcnt(0)`×3–4 per tile) | 2.1, **all partial** (`vmcnt(10..39)`) |
| outstanding loads | drains to 0, 3–4× per k-tile | **13–39 sustained** ("pf3" = 3 tiles deep) |
| MFMA density | 7–8% | 8% |

**The mix is nearly at parity — the gap is pipeline DEPTH alone**, which corroborates
the activation-arms triple falsification ("issue/latency-bound"): same instruction diet,
but aiter never lets the memory pipe empty.

## 3. GEMM: `d_gemm_t` k-loop vs `bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk`

| metric | plow (64–72 insts) | aiter (703 insts) |
|---|---|---|
| MFMA | 8 | 240 |
| VALU / MFMA | 1.6 | **0.0 — the loop contains no VALU at all** |
| gload / MFMA | 0.4 | 0.55 |
| ds_read / MFMA | 1.0 | 0.5 |
| waitcnt | 4–7 per 8 MFMA (partial `lgkmcnt(2)` + full `lgkmcnt(0)`) | **6 per 240 MFMA — a single `vmcnt(22) lgkmcnt(0)` shape** |
| setprio | 8/iter (occ-2 ping-pong, already `if constexpr(PRIO)`-parameterized) | 0 |
| MFMA issue density | 11–12% | **34%** |

aiter's GEMM loop does ALL addressing on the scalar unit (199 SALU, zero VALU) with 22
loads perpetually outstanding and one wait per unroll body.

## Cheap-lever verdicts (this branch)

- **`PLOW_MOE_PF_SCHED` (sched_group_barrier pipeline shaping) — NULL BY CONSTRUCTION,
  landed default-off with a do-not-retry note at the site.** The barriers moved exactly
  24 instructions (12 ds_read_b128 + address adds) and changed no waitcnt: the waits are
  dataflow-forced (loop-head loads feed the same iteration's single-buffer LDS commit),
  so no scheduler hint can create aiter's depth. GPU A/B skipped — a 24-instruction move
  in a 69k-line object is asm-provably noise.
- **GEMM setprio removal — not attempted**: already `if constexpr (PRIO)` with the
  occ-2 ping-pong rationale documented in op_gemm.h; the in-tree default was measured.
- No other HIP-expressible lever survives the diff: the flash's VALU is algorithmic
  (softmax; SMX already split it), and partial-waitcnt discipline is not reachable from
  HIP when the dependency graph forces full drains.

## Ranked asm-class techniques (input to the asm-MoE-restructure decision)

1. **Sustained deep outstanding loads** (13–39; `pf3` multi-tile register pipeline) with
   ONE partial waitcnt per phase — requires manual waitcnt control or a restructured
   multi-buffer register pipeline; the single decisive difference in the MoE loop.
2. **Scalar-unit addressing** — aiter's GEMM inner loop is VALU-free; ours burns 1.6
   VALU/MFMA on address math that could be SGPR-resident.
3. **Streamed operand global→VGPR** (no LDS round trip) for KV/B — 4 ds_writes/iter in
   aiter's fmha vs our 24; halves LDS pressure and barrier count.
4. **Huge unroll bodies** (703–3621 insts) with zero loop-carried stalls — amortizes the
   loop overhead our 64–232-inst bodies pay every tile.
5. **Lean softmax interleave** — 6.7 VALU/MFMA including softmax vs our 19–21 in the KV
   body; needs fragment-map/register-layout co-design, not local edits.

Artifacts: disassemblies + loop extractor under jobs tmp `asmagent/`; probe committed at
`perf-data/plow-gfx942/probes/asm_loops.py`.
