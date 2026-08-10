# Gemma-4-26B-A4B — single-block plow-vs-vLLM DECODE baseline + tune verdict + K-split probe

GPU: **H100 NVL** (sm_90a, 132 SMs, HBM3 ~3350 GB/s spec), bf16, M=1 unless noted.
All GPU runs under `gpulease` (rc=0, uncontended). Framework:
`scripts/block_layer_bench.py` (whole block) + new `scripts/block_op_bench.py`
(per-op, both phases, ctx sweep) driving vLLM 0.25.1's **real** `Gemma4DecoderLayer`.

The 26B-A4B block runs BOTH a dense MLP **and** a 128-expert/top-8 MoE in parallel
(`Gemma4DecoderLayer.forward`), plus ~10 RMSNorms + RoPE. Config verbatim from
`perf-data/block-configs/gemma4-26b-a4b-moe.json` (hidden 2816, 16Q/8KV, hd 256,
inter 2112, 128 experts top-8, moe_inter 704, sliding-window layer 0).

## TL;DR verdict

1. **Per-op target (vLLM, measured).** The GEMV family is the whole game at decode.
   The single biggest kernel is **`moe_experts` (56.5 us)**, then **`qkv_proj`
   (21.8 us)**, `attn` (23 us, latency-bound), `o_proj` (15.7), `mlp_gate_up`
   (13.8), `mlp_down` (10). vLLM runs these projections at **36–63 % of HBM peak**.
2. **plow gap is uniform, not one bad op.** plow's decode body runs at **~21 % of
   HBM peak** (campaign roofline, 816 GB/s), i.e. ~**2.0–2.4× behind vLLM on every
   projection**. This is the shared 1-block/SM occupancy ceiling, so the kernel
   agent should attack the **GEMV occupancy/bandwidth** globally; among individual
   ops, **`moe_experts` is the largest single prize**.
3. **Tune system is NOT a lever here.** Ran it. On sm_90a all 3 GEMM opcodes alias
   to ONE body (`d_gemm_sm90`, fixed 128×128×64 wgmma tile, compile-time macro);
   "there is nothing to rank". For **decode GEMV the tuner has no kernel at all** —
   it errors. Tier is always `portable` (analytical); no measurements, can't make
   any. It cannot touch the memory-bound decode path.
4. **K-split is refuted for the dense projections too.** Standalone probe:
   down_proj **−25.5 %**, o_proj **−15.5 %**, lm_head **−14.6 %** vs best N-split.
   The real lever the probe exposes is **blocks-per-SM (occupancy)**, and plain
   N-split exploits it better than K-split at every block count.

---

## Task 1 — per-op single-block DECODE baseline (the target to beat)

Measured with `block_op_bench.py`: each op captured into its own CUDA graph
(erases launch/python overhead — the regime real graphed decode runs in),
replayed with an L2 flush between replays so every op reads its weight **cold**
from HBM. `moe_experts` verified flush-insensitive (56.5 us at both 96 MB and
256 MB flush → not an L2-residency artifact). Achieved GB/s = read_bytes / us.

### vLLM `Gemma4DecoderLayer`, DECODE M=1, ctx=1024 (H100 NVL, bf16)

| op | us | GB/s | % HBM peak | class | notes |
|---|---:|---:|---:|---|---|
| **moe_experts** (top-8) | **56.5** | 1684 | 50 % | GEMV | biggest single kernel |
| attn (FlashDecode) | 23.0 | 365 | — | attn | latency-bound at ctx≤4k (~23 us flat) |
| **qkv_proj** | **21.8** | 2119 | 63 % | GEMV | most efficient projection |
| o_proj | 15.7 | 1470 | 44 % | GEMV | |
| mlp_gate_up | 13.8 | 1724 | 51 % | GEMV | |
| mlp_down | 10.0 | 1194 | 36 % | GEMV | small-K, lower %HBM |
| moe_router | 28.8 | 25 | — | tiny GEMV | triton routing kernel, latency-bound |
| rmsnorm (one of ~10) | 18.5 | — | — | norm | latency floor, not BW |
| **SUM of ops** | **188** | | | | |
| **whole block (cudagraph)** | **356** | | | | `block_layer_bench`, this session |

**Reading it.** GEMV family (qkv+o+gate_up+down+router+moe_experts) = **147 us =
78 % of the per-op sum**; FlashDecode ≈ 12 %; one norm ≈ 10 %. This **confirms the
C1 whole-model split (GEMV ≈ 84 % of body, FlashDecode ≈ 16 %) at block
granularity**, and splits the GEMV by projection (table above).

**Why sum (188) < whole block (356).** The whole block launches ~10 separate
RMSNorm/RoPE/residual/combine kernels, each paying a ~18 us launch-latency floor
at M=1 (~180 us total). plow **fuses these into the packet** (no per-op launch),
so this overhead is a vLLM artifact of running one isolated block, not a real
per-block cost in a full-model cudagraph. Corollary: **isolated single-block
decode over-states the norm floor** — the transferable, optimisation-relevant
numbers are the per-op GEMV/attn rows, not the whole-block us. (Cross-check: the
full-model per-layer floor is vLLM 4.833 ms/30 ≈ 161 us vs plow 9.34 ms/30 ≈
311 us = the same ~1.9× gap.)

### plow side (from measured campaign data, `gemma26b-h100-beat-vllm-campaign.md`)

A live block-level plow per-op trace (`-DPLOW_NV_TRACE=1` cubin + `block_run`
`trace_summary`) was **not** re-run here — it needs a trace cubin + single-block
asset build, and the decode per-op picture is already measured whole-model:

- decode body = **816 GB/s = 20.9 % of HBM peak** (roofline, triply corroborated).
- op-class: body 68 % | inter-op counter-gate WAIT **29 %** | signal 3 %; within
  body GEMV ≈ 84 %, FlashDecode ≈ 16 %. Both the 21 %-of-peak and the 29 % gate
  stall are symptoms of the **1 block/SM = 12.5 % occupancy** megakernel.

### plow-vs-vLLM per-op ratio (the concrete target)

| projection | vLLM % HBM (measured) | plow % HBM (roofline) | plow gap |
|---|---:|---:|---:|
| qkv_proj | 63 % | ~21 % | **~3.0×** |
| mlp_gate_up | 51 % | ~21 % | ~2.4× |
| moe_experts | 50 % | ~21 % | ~2.4× |
| o_proj | 44 % | ~21 % | ~2.1× |
| mlp_down | 36 % | ~21 % | ~1.7× |

The gap is **roughly uniform** because it is the shared megakernel occupancy
ceiling, not a per-kernel maturity problem. **Prioritise the GEMV occupancy/BW
globally** (occ2/occ3, cp.async row-staging); the largest single-op win is
`moe_experts`.

### PREFILL per-op (same harness, `--phases prefill`) — the tensor-core target

vLLM `Gemma4DecoderLayer`, PREFILL, M=T (whole-block prefill = 2.45 ms @1024):

| op | ctx1024 us / TFLOP·s | ctx4096 us / TFLOP·s |
|---|---|---|
| moe_experts | 1058 / 92 | 2396 / 163 |
| qkv_proj | 124 / **380** | 500 / 378 |
| mlp_gate_up | 52 / 465 | 266 / 366 |
| o_proj | 47 / 497 | 246 / 383 |
| mlp_down | 29 / 415 | 131 / 373 |
| attn | 55 | 252 |

Prefill dense projections hit **~370–500 TFLOP/s** (vLLM cuBLAS/FA). This is the
number plow's wgmma `d_gemm_sm90` (128×128×64) must beat — and it is exactly what
the tune system claims to select (Task 2). Single-block prefill sweep means a new
model's prefill GEMM target is available **without a full-model run**.

---

## Task 2 — does the plow tune system help? **No, not on this GPU/path.**

Read `crates/plowc/src/{tune,tuned}.rs`, built
`plowc`, and RAN the tuner on H100 NVL. What it tunes: **prefill dense-GEMM tile
selection** (BM×BN×BK) via a probed capability inventory + analytical cost model
(+ optional measured DB). It does **not** benchmark (by design) and does **not**
model decode GEMV.

**Inventory probe (`plowc tune --gpu "H100 NVL" --profile prefill_dense`):**
```
executable kernels (3):
  8   PLOW_DOP_GEMM        tile 128x128x64   dispatched
  15  PLOW_DOP_GEMM_MED    tile 128x128x64   dispatched
  14  PLOW_DOP_GEMM_SMALL  tile 128x128x64   dispatched
aliases: sm_90a:d_gemm_sm90  <-  GEMM, GEMM_MED, GEMM_SMALL
  "on NVIDIA the tile is a compile-time macro per interpreter object, so the
   real tuning axis is which object is built, not which opcode is emitted."
```
All three GEMM opcodes **alias to one body** with **one fixed 128×128×64 wgmma
tile**. `select` for every real 26B prefill shape (q/kv/o/gate_up/down/moe) returns
the same kernel: *"3 opcodes share one implementation; there is nothing to rank"*,
tier `portable`.

**Decode GEMV: the tuner has nothing.** `--profile decode_dense --shape 1,4096,2816`:
```
op: dense matmul 1x4096x2816 (Gemv)
ERROR: no kernel in the decode_dense profile on nvidia/sm_90a/h100-nvl can run
       DenseMatmul 1x4096x2816; compiling one anyway would emit an opcode with
       no dispatch arm
```
`--status`: *no kernel measurements for this cell; selection will use the
analytical model and report tier `portable`.*

**Verdict.** The tune system is **not a real lever for 26B on H100**:
- Decode (the 84 %-of-body GEMV, memory-bound): **untouched** — no decode-GEMV
  kernel in the inventory; the tuner errors. Zero effect on the decode gap.
- Prefill (tensor-core GEMM): the tile is **fixed by which cubin is built**
  (compile-time macro `PGM90_*` in `op_gemm_sm90.cuh`), the packet's tile fields
  are ignored by `d_gemm_sm90`, and the 3 opcodes are aliases → **emitting a
  different opcode changes nothing at runtime**. Tuning cannot pick a better tile
  because there is only one.
- No measurements exist and the tuner cannot produce any (`plowc tune` doesn't
  benchmark). `--tuning-db` / `--no-tuning` produce byte-identical output.
- Bonus finding: `gemma4.rs::pick_tile` hardwires `hwspec::lookup("MI350X")` — the
  emitter ranks tiles against the **AMD gfx950** cost model regardless of the real
  target (the plan's gap #2, "not yet the implementation reality"). Irrelevant on
  sm_90a only because the one buildable tile wins trivially.

The only actual "tuning axis" on NVIDIA is **rebuilding the interpreter cubin with
different `-D` macros** (e.g. `PLOW_NV_FORCE_MINBLK=2` → occ2, the campaign's
measured +6.2 %). That is a build knob, **outside** the tune system.

---

## Task 3 — K-split for non-headnorm projections: **refuted, quantified**

Standalone probe `runtime/nvidia/experiments/ksplit_gemv.cu` (sm_90a). Compares
**N-split** (one-warp-per-output-row = production `gemv_rows`) vs **K-split**
(S warps/row, each dots K/S, atomicAdd reduce) at the real down/o_proj shapes,
sweeping blocks/SM. Weight-BW = N·K·2 / time (reduce cost *excluded* from the
numerator, so K-split is if anything flattered).

| shape | best N-split | best K-split | K-split delta |
|---|---|---|---|
| down_proj (N2816,K2112) | **1536 GB/s** (46 % HBM, 3 blk/SM) | 1144 (S=2) | **−25.5 %** |
| o_proj (N2816,K4096) | **1952 GB/s** (58 %, 3 blk/SM) | 1650 (S=2) | **−15.5 %** |
| lm_head ctrl (N32000,K2816) | **2302 GB/s** (69 %, 4 blk/SM) | 1967 (S=2) | **−14.6 %** |

**K-split loses everywhere.** The atomic reduce + K-slice tail imbalance costs
15–25 % and never buys back BW, because N=2816 at nblk=132 **already puts one block
on every SM** — the binding limit is warps-per-SM, which K-split does not raise.

**What the probe *does* show is the real lever — occupancy (blocks/SM):**

| down_proj N-split | 1 blk/SM | 2 blk/SM | 3 blk/SM | 4 blk/SM |
|---|---|---|---|---|
| GB/s (% HBM) | 1186 (35 %) | 1302 (39 %) | **1536 (46 %)** | 1314 (39 %) |

Going 1→3 blocks/SM lifts achieved BW 35 %→46 % (down) and 45 %→58 % (o_proj) —
consistent with the campaign's occ2 (+6.2 %). **Recommendation to the kernel
agent: pursue higher blocks/SM with N-split (occ2/occ3 + cp.async row-staging);
drop K-split** for the dense projections (as it was already dropped for the small
MoE-expert shapes in `gemv_dimspec.cu`, −30 %).

Caveat: this isolated probe (x resident in smem, GPU to itself) reaches higher
%HBM than plow's in-megakernel GEMV, which shares the grid with all other ops.
The **N-split-vs-K-split delta is the valid, transferable result**; the absolute
%HBM is an upper bound for a dedicated kernel.

---

## Harness additions (this task)

- **`scripts/block_op_bench.py`** — new per-op single-block harness. Reuses
  `block_layer_bench.py`'s layer construction (import, no drift). `--phases
  decode,prefill --ctx 1024,4096`: per-op device time + GB/s + TFLOP/s via
  CUDA-graph-per-op + L2-flush. **This is the "no full-model run" tuning path the
  ask calls for** — onboarding a new model's per-op decode AND prefill targets
  now needs only a single-block run (a JSON config, no checkpoint/HF download).
- **`runtime/nvidia/experiments/ksplit_gemv.cu`** — the K-split vs N-split probe.
- JSON: `/dev/shm/block-op/26b-op.json` (per-op sweep).

### Bottom line for the kernel agent
Attack the **decode GEMV occupancy/bandwidth globally** (all projections are ~2×
behind vLLM at the *same* 21 %-of-peak ceiling; `moe_experts` is the single
biggest op at 56 us). The **tune system won't help** (no decode-GEMV kernel; prefill
tile is a fixed compile-time macro). **K-split won't help** (−15…−25 %); the lever
is **blocks/SM (occ2/occ3) + cp.async row-staging with the existing N-split map**.
