# Gemma-4-26B-A4B fp8 PREFILL — single-block, H100 NVL (sm_90a)

GPU: **H100 NVL** (sm_90a, 132 SMs). Base commit `ed9e7dd`. fp8 = e4m3 (w8a8).
All existing fp8 perf-data is sm_120/RTX; this is the first **H100 sm_90a** fp8
prefill characterisation. Two independent measurements + a source/emit audit:

1. **Capability audit** — is the campaign's "fp8 has NO prefill program" a KERNEL
   gap, an EMITTER gap, or a served-asset gap? (emit + byte-diff, no GPU)
2. **vLLM fp8 per-op prefill baseline** — `block_op_bench.py --quant fp8` (real
   `Gemma4DecoderLayer`, L2-flushed CUDA-graph-per-op), ctx 1024/4096.
3. **plow fp8 w8a8 wgmma micro-bench** — the PRODUCTION `d_gemm_w8a8_sm90` /
   `d_gemm_glu_w8a8_sm90` bodies at the real 26B prefill shapes, standalone.

---

## TL;DR — the capability gap is REAL for the served asset, but it is NOT a kernel gap

- The fp8 wgmma prefill **KERNELS EXIST and are LIVE in the served prefill cubin**
  (`d_gemm_w8a8_sm90` op_gemm_sm90.cuh:321, `d_gemm_glu_w8a8_sm90` :431).
- The plowc **EMITTER CAN emit a full fp8 prefill program** — but only on the
  **w8a8** path (`PLOW_W8A8=1`): 5 prefill buckets, GemmFp8/GemmGluFp8 + grouped-MoE.
- The **served fp8 pkt has ZERO prefill programs.** It was built plain-fp8
  (`PLOW_FP8=1`, w8a16), and for a **MoE model** plain-fp8 grouped-MoE prefill is
  unimplemented → the emitter **drops the entire prefill bucket set** and ships a
  **decode-only** packet. Proven byte-for-byte (below).
- Standalone, the w8a8 wgmma prefill kernel **beats plow's own bf16 wgmma prefill
  by 1.7–1.9×** (342 vs 186 TF/s on qkv @4096) and reaches **15–18 % of the H100
  fp8 peak** — but it is **~0.6–0.8× of vLLM fp8** (which hits 54–67 % of bf16-equiv
  rate). This is pure **headroom**: none of it is wired into the served pkt.

---

## Task 1 — Capability-gap resolution (definitive)

**Question the campaign left open:** kernel absent, or emitter doesn't emit it?
Answer, three parts:

### (a) Kernel present? — YES, and live in the served cubin
`d_gemm_w8a8_sm90` / `d_gemm_glu_w8a8_sm90` are Hopper wgmma e4m3 (QGMMA)
prefill bodies, drop-in for the mma.sync `d_gemm_w8a8` under `#if PLOW_NV_HOPPER`
(op_gemm.cuh:1150, :1232). `scripts/build_sm90a_cubin.sh` builds
`interp_sm90a_pf.cubin` with them **live** — its own comment: *"the LIVE Hopper
wgmma GEMM arms (d_gemm_sm90 / **d_gemm_w8a8_sm90**)"* own the prefill REG=255
ceiling. So the served prefill cubin **can execute** a w8a8 prefill GEMM opcode.
(Caveat: the DEFAULT cubin runs the **w8a16** arm of the shared `GEMM_FP8` opcode —
bf16 activation × e4m3 weight; the true **w8a8** arm needs a `PLOW_NV_W8A8=1`
cubin, gemma4.rs:6055-6060. Neither is exercised by the served pkt — see (c).)

### (b) Emitter emits it? — YES on the w8a8 path; the prefill program is gated on it
The dense-projection prefill arm emits an fp8 tiled-GEMM opcode when `fp8`
(gemma4.rs:1848-1868): `pick_tile → GemmFp8 / GemmMedFp8 / GemmSmallFp8`, and the
GLU arm emits `GemmGluFp8` (:2736). So **dense** fp8 prefill emission is present.
**But the packet is a MoE model**, and whether ANY prefill bucket is emitted hinges
on `moe_pf` (gemma4.rs:6148-6163):

```
let moe_pf = c.moe && (!fp8 || w8a8) && env("PLOW_MOE_PREFILL") != "0";
for &t in &buckets { if c.moe && !moe_pf { break; } ... emit prefill bucket ... }
```

- plain fp8 (`PLOW_FP8=1`, `w8a8=false`): `moe_pf = moe && (!true || false) = false`
  → the bucket loop **breaks on iteration 0** → **no prefill program emitted at all**.
- w8a8 (`PLOW_W8A8=1`): `moe_pf = true` → all 5 prefill buckets emitted, incl. the
  grouped-MoE w8a8 prefill (ops 81/82, comment gemma4.rs:6145-6147).

The blocker is specifically **grouped-MoE prefill for plain fp8 (w8a16 dequant),
which is unimplemented**; the dense fp8 prefill GEMM would emit fine. Because a MoE
model cannot ship a prefill program without its expert prefill, the whole set drops.

### (c) Program in the SERVED pkt? — NO (decode-only), proven byte-for-byte
Re-emitting exactly as `scripts/build_gemma4_h100_assets.sh` does
(`PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 PLOW_FP8=1 gemma4 <model> 8192 … 132`):

| build | prefill programs | decode | pkt bytes |
|---|---:|---|---:|
| **plain fp8 (w8a16) — SERVED** | **0** (loop breaks: MoE + !moe_pf) | prog0 T=1 | 26 720 620 |
| w8a8 (`PLOW_W8A8=1`) | 5 (T=128…4096) | prog5 T=1 | — |
| bf16 (reference) | 5 (T=128…4096) | prog5 T=1 | — |

The plain-fp8 emit is **byte-identical (26 720 620 B) to the served
`/workspace/assets/plowrt-26b/fp8/model.pkt`** → the served fp8 asset contains
**exactly one program, the T=1 decode program.** A `plowrt serve` of it consumes a
prompt through the **decode** path (no w8a8 tensor-core prefill runs); it does not
"fall back to bf16" and it does not error — there simply is no prefill program to
dispatch.

**Verdict on the campaign claim.** *"fp8 has NO prefill program"* is **TRUE for the
as-built plain-fp8 asset**, but the nuance the campaign missed: it is **not** a
kernel/emitter absence. The w8a8 wgmma prefill kernels exist, are live in the cubin,
and the emitter ships a complete 5-bucket fp8 prefill program under `PLOW_W8A8=1`.
The gap is (i) the served asset was built w8a16, and (ii) plain-fp8 **grouped-MoE**
prefill is unimplemented, which forces a MoE model to decode-only. Wiring fp8
prefill = build+serve the `PLOW_W8A8=1` packet against a `PLOW_NV_W8A8=1` prefill
cubin (both already exist as build knobs).

---

## Task 2 — vLLM fp8 per-op PREFILL baseline (the target)

`block_op_bench.py --phases prefill --quant fp8` (online e4m3, dynamic act-scale),
real `Gemma4DecoderLayer`, H100 NVL. TFLOP/s = 2·M·N·K / device-us.

| op | ctx1024 us / TF/s | ctx4096 us / TF/s |
|---|---|---|
| qkv_proj | 70.6 / **669** | 347 / **544** |
| o_proj | 42.4 / 557 | 154 / **613** |
| mlp_gate_up | 45.1 / 540 | 215 / 454 |
| mlp_down | 34.3 / 355 | 127 / 382 |
| moe_experts | 349 / 279 | 1315 / 296 |
| moe_router | 70.0 / 10.6 | 229 / 12.9 |

vLLM fp8 dense projections hit **~540–670 TF/s** — and are **faster than vLLM bf16**
(same harness, block-decode-baseline doc: qkv 380, o_proj 497, gate_up 465, down
415 TF/s). fp8 is a real win for vLLM's prefill; it is the number plow must beat.
(attn skipped — AttributeError in the isolated fp8 attn wrapper; attn is not an fp8
GEMM target. Absolute us are per-op isolated, so they over-count vs a fused block.)

---

## Task 3 — plow fp8 w8a8 wgmma PREFILL micro-bench (standalone)

Probe `runtime/nvidia/experiments/prefill_fp8_wgmma.cu`: the production
`d_gemm_w8a8_sm90` / `d_gemm_glu_w8a8_sm90` (via `d_gemm_w8a8` / `d_gemm_glu_w8a8`
under `PLOW_NV_HOPPER`), 132 blocks = 1/SM (the megakernel regime), vs plow's OWN
bf16 wgmma (`d_gemm_sm90` / `d_gemm_glu_sm90`) at the same shapes. Peaks: fp8 e4m3
**1979 TF/s**, bf16 **989.5 TF/s**.

### ctx 4096 (M=4096) — the tensor-core regime

| shape (26B) | plow bf16 TF/s (%bf16pk) | plow **w8a8** TF/s (%fp8pk / %bf16pk) | vs plow bf16 | vs vLLM fp8 |
|---|---:|---:|---:|---:|
| qkv N6144 K2816 | 186 (18.8%) | **342** (17.3% / 34.6%) | **1.84×** | 342/544 = 0.63× |
| o_proj N2816 K4096 | 206 (20.8%) | **357** (18.1% / 36.1%) | 1.73× | 357/613 = 0.58× |
| down N2816 K2112 | 189 (19.1%) | **310** (15.7% / 31.4%) | 1.64× | 310/382 = 0.81× |
| gate_up N2112 K2816 (GLU) | 172 (17.4%) | **302** (15.3% / 30.6%) | 1.75× | 302/454 = 0.67× |

### ctx 1024 (M=1024)

| shape | plow bf16 TF/s | plow **w8a8** TF/s (%fp8pk) | vs plow bf16 | vs vLLM fp8 |
|---|---:|---:|---:|---:|
| qkv | 207 | 351 (17.8%) | 1.69× | 351/669 = 0.53× |
| o_proj | 153 | 284 (14.4%) | 1.86× | 284/557 = 0.51× |
| down | 141 | 226 (11.4%) | 1.61× | 226/355 = 0.64× |
| gate_up (GLU) | 108 | 192 (9.7%) | 1.77× | 192/540 = 0.35× |

**ptxas (standalone, sm_90a, `-Xptxas -v`):** w8a8 GEMM **134 regs, 0 spills**;
w8a8 GLU **165 regs, 0 spills** (bf16: 134 / 168). The **~680-spill** concern
(op_gemm.cuh:519-524) is a **megakernel** effect: it is measured on the COMBINED
tree (all op_* forks + the interpreter, at the 255-reg ceiling), where FORK_GLU adds
~680 spills on top of the plain fork's ~910 and spills OTHER arms (+252 attn, +261
interp). Isolated, the kernel is spill-free — the register pressure, not the kernel,
is the cost. This is why FORK_GLU defaults OFF and why fp8 prefill wants TMA +
setmaxnreg warp-spec (the real fix, plans/h100-hopper-optimization.md).

**Accuracy (fp8 is lossy — sanity only):** w8a8 vs bf16-kernel output relL2 =
**8.3–9.3 %** (GEMM) / **13.4 %** (GLU). Unscaled worst case (scale≡1, values in
[−0.5,0.5] → e4m3's coarse mantissa near small magnitudes); production per-tensor /
per-channel calibration that uses e4m3's full range cuts this substantially. It is a
valid fp8 result, not a mis-decode.

---

## VERDICT

- **(a) present & usable in the served pkt?** **NO.** The served fp8 asset is
  byte-identically the plain-fp8 (w8a16) build = **decode-only, 0 prefill programs**.
  The kernels exist and are live in the cubin, and a `PLOW_W8A8=1` packet WOULD ship
  5 fp8 prefill buckets — but that packet is not built/served. Capability gap: real,
  but it is an emit-config + unimplemented-w8a16-MoE-prefill gap, not a kernel gap.
- **(b) faster than plow bf16 prefill?** **YES, standalone: 1.6–1.9×** (342 vs 186
  TF/s qkv @4096; 15–18 % of fp8 peak vs bf16's 19–21 % of bf16 peak). The kernel
  works and pays off — this is the headroom if wired.
- **(c) competitive with vLLM fp8?** **NO, ~0.6–0.8× at ctx4096** (worse at 1024:
  0.35–0.64×). vLLM fp8 runs dense projections at 540–670 TF/s (27–34 % of fp8 peak);
  plow's standalone w8a8 wgmma is 300–360 TF/s (15–18 %). Even the *best-case*
  standalone kernel — before the megakernel occupancy/spill tax — is ~1.5× behind
  vLLM fp8, i.e. fp8 does **not** close the ~2× prefill gap on its own; it needs the
  same Hopper TC pipeline work (TMA + warp-spec) the bf16 prefill needs.

---

## Reproduction

```bash
# (1) capability audit — CPU only, no gpulease
M=/workspace/models/gemma-4-26B-A4B-it
PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 PLOW_FP8=1            ./target/release/gemma4 $M 8192 /tmp/fp8.pkt 132   # 1 prog (decode)
PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 PLOW_FP8=1 PLOW_W8A8=1 ./target/release/gemma4 $M 4096 /tmp/w8a8.pkt 132  # 5 prefill + decode
cmp /tmp/fp8.pkt /workspace/assets/plowrt-26b/fp8/model.pkt   # identical

# (2) vLLM fp8 per-op prefill
CFG=/root/plow/.claude/worktrees/block-baseline-harness/perf-data/block-configs/gemma4-26b-a4b-moe.json
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease blkop bash -lc \
  "export PATH=/workspace/venvs/vllm-blk/bin:\$PATH; \
   export BLB_DIR=/root/plow/.claude/worktrees/block-baseline-harness/scripts; \
   python scripts/block_op_bench.py $CFG --phases prefill --ctx 1024,4096 --quant fp8"

# (3) plow w8a8 wgmma micro-bench (NOTE: -gencode compute_90a; -arch=sm_90a downgrades to sm_90 → wgmma rejected)
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -gencode arch=compute_90a,code=sm_90a -O3 -Xptxas -v \
  -I runtime/common -I runtime/nvidia -DPLOW_NV_HOPPER=1 -DPGM90_FORK_GLU=1 \
  -o /tmp/pfp8 runtime/nvidia/experiments/prefill_fp8_wgmma.cu
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat gpulease pfp8 /tmp/pfp8
```

Artifacts: probe `runtime/nvidia/experiments/prefill_fp8_wgmma.cu`; vLLM JSON
`/dev/shm/block-op/26b-fp8-pf.json`. No production interpreter/emitter source modified.
