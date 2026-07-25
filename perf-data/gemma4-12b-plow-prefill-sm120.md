# gemma-4-12B bf16 prefill on sm_120 (RTX PRO 6000 Blackwell, 188 SMs)

plow chunked PREFILL through the persistent sm_120 interpreter (`gemma4_sm120_chat`,
`PLOW_PREFILL=1`, `PLOW_UNISEG=1`), batch 1, TP1, deterministic prompts. `prefill_ms`
is PURE prefill wall time (KV build for `n_prompt` tokens, EXCLUDING the first decode
token); vLLM's `ttft_ms` (comparison column) includes prefill + first-decode + scheduling,
so `plow_prefill_ms` is the strict lower bound of an equivalent TTFT. Numbers are the
literal source values; the machine-readable siblings are `*.json` (this file mirrors them,
it is not re-transcribed by `consolidate_perf.py`).

## Campaign progression (prefill_ms)

| ctx    | T3 cp.async pipe | T4 mma P·V | **T5 cp.async KV-stream** | vLLM bf16 TTFT | T5 vs vLLM | T5 vs T4 |
|--------|------------------|------------|---------------------------|----------------|-----------|----------|
| 4k     | 849.44           | 612.55     | **513.93**                | 323.5          | 1.59x     | −16.1%   |
| 16k    | 5419.74          | 3342.18    | **2169.25**               | 1502.3         | 1.44x     | −35.1%   |
| 32k    | 16613.76         | 9424.35    | **5050.37**               | 2815.2         | 1.79x     | −46.5%   |
| 64k    | 60311.62         | 34301.74   | **12928.96**              | 8469.7         | 1.53x     | −62.3%   |
| 128k   | 288701.20¹       | 196806.86¹ | **37223.01**¹             | 16271.0        | 2.29x     | −81.1%   |

¹ n_prompt=131000 (p_131000.ids); all other rows are exact powers of two.
(Earlier rungs G5/T1/T2 are in the JSON; T3→T5 shown here. The T5 A/B "base" build
— `PLOW_NV_FA_PIPE=0` — reproduced T4's 128k within 0.14% in the SAME session.)

**T5 closes the 128k gap from 12.11× to 2.29× and beats T4 at every rung — the win
GROWS with ctx** (−16%@4k → −81%@128k) because the O(ctx²) full-attention (hd512)
layers dominate long-ctx prefill and, under T4's synchronous staging, were almost
pure HBM-wait between tiny mma steps. The cp.async KV-stream pipeline overlaps the
K/V load with the QK/softmax/P·V compute and recovers most of it. Still bf16
1.4–2.3× behind vLLM's tuned paged FA-3; the biggest remaining rung is 128k.

## T6-L2 fp8 (w8a16) PREFILL GEMM (2026-07-19) — MEASURED NEGATIVE for speed

`d_gemm_fp8` / `d_gemm_glu_fp8` (op_gemm.cuh): e4m3 weight, bf16 activation, **dequant-to-bf16
-in-smem** (fp8 tile cp.async'd at HALF the bytes, expanded to bf16 in smem, then the EXISTING
bf16 mma inner loop runs unchanged; per-output-channel dequant scale in the epilogue — it factors
out of the K reduction). Emitter emits GEMM_FP8/MED/SMALL_FP8 + GEMM_GLU_FP8 under `PLOW_FP8=1`;
the now-dead bf16 projection weights are elided in fp8 mode (packet 32.3 → **12.0 GiB**).

| ctx  | bf16 (T5) | **fp8 w8a16** | Δ vs bf16 | vLLM fp8 TTFT | plow fp8 vs vLLM fp8 |
|------|-----------|---------------|-----------|---------------|----------------------|
| 4k   | 513.93    | **516.02**    | +0.4%     | 244.71        | 2.11×                |
| 16k  | 2169.25   | **2245.95**   | +3.5%     | 1220.76       | 1.84×                |
| 32k  | 5050.37   | **5207.31**   | +3.1%     | 2438.73       | 2.14×                |
| 64k  | 12928.96  | **13265.06**  | +2.6%     | 7663.76       | 1.73×                |
| 128k | 37223.01¹ | **37891.30**¹ | +1.8%     | 15520.48      | 2.44×                |

**Verdict: w8a16 dequant-in-smem is +0.4…+3.5% SLOWER than bf16 at every rung.** Root cause: the
prefill GEMM is **compute-bound**, not weight-bound — large M (128…8192 rows) reuses each weight
across many output rows, so the mma (not the weight stream) is the wall; halving DRAM weight
traffic buys nothing while the dequant convert pass adds overhead. This is the OPPOSITE of decode
(M=1, bandwidth-bound, where the SAME fp8 weights win −10% TPOT). Because plow fp8 prefill is
slightly slower than its own bf16 while vLLM fp8 is FASTER than its bf16 (true fp8 tensor cores),
fp8 makes plow's RELATIVE gap vs vLLM WORSE (1.7–2.4× vs bf16's 1.4–2.3×). **The real prefill fp8
lever is w8a8** (fp8 `mma.sync.m16n8k32`, 2× tensor throughput + per-row activation quant), for
which this w8a16 path is the correctness-complete foundation (opcodes, dispatch, emitter, oracle).

**Gates:** A — 6 new fp8 GEMM/GEMM_GLU oracle cases vs a dequantized-f32 reference (per-row
amax/448 e4m3) PASS at relL2 6.8e-5…1.7e-4 (tighter than gemv_fp8's 1.6e-3: the f32 mma accumulate
beats the warp-tree); bf16 GEMM/flash unchanged; wave64 negctrl FAILs flash 20/20. B — fp8 prefill
first token AGREES with bf16 prefill at every ctx (p_short 236761, p_4k 874; device==host argmax
AGREE at 32k/64k/128k). D — decode gemma cubin BYTE-IDENTICAL; ptxas 238 regs / occ-1 UNCHANGED.

## T8-w8a8 PREFILL GEMM (2026-07-19) — MEASURED POSITIVE — first prefill lever to BEAT plow bf16

The compute-bound fix T6 pointed at, now realized. w8a8 uses TRUE fp8 tensor cores
(`mma.sync.m16n8k32.e4m3`, 2× the bf16 rate) with BOTH operands e4m3: the weight is the
per-output-channel e4m3 twin (T6, unchanged), the **activation** is per-M-row e4m3 quantized by a
new `QUANT_FP8` pass. The emitter (`gemma4.rs`, `PLOW_W8A8=1`, requires `PLOW_FP8=1`) emits ONE
shared `DevOp::QuantFp8` per activation site — qkv input, o_proj input, gate/up input, down input
(4/layer, 192 total) — and re-points `GEMM_FP8`/`GEMM_GLU_FP8` to `t1=xq` (fp8 activation) + `t3=a_scale`.
The shared single quant is **required for correctness**: a per-proj quant would race the xq buffer
that the sibling GEMMs read. lm_head stays bf16 (no fp8 embed twin). The SAME opcodes serve T6
w8a16; the interp cubin picks the kernel by `PLOW_NV_W8A8`, so a w8a8 pkt MUST run a `PLOW_NV_W8A8=1`
prefill cubin. `PLOW_W8A8` unset ⇒ emission byte-identical to bf16 and to the committed fp8-only pkt.

| ctx  | plow bf16 | **plow w8a8** | Δ vs bf16 | vLLM fp8 TTFT | w8a8 vs vLLM-fp8 | (plow bf16 vs vLLM bf16) |
|------|-----------|---------------|-----------|---------------|-------------------|---------------------------|
| 4k   | 513.97    | **429.41**    | **−16.5%**| 244.71        | 1.75×             | 1.59×                     |
| 16k  | 2169.97   | **1918.41**   | **−11.6%**| 1220.76       | 1.57×             | 1.44×                     |
| 32k  | 5058.93   | **4554.95**   | **−10.0%**| 2438.73       | 1.87×             | 1.80×                     |
| 64k  | 12979.20  | **11964.73**  | **−7.8%** | 7663.76       | 1.56×             | 1.53×                     |
| 128k | 37351.31¹ | **35485.36**¹ | **−5.0%** | 15520.48      | 2.29×             | 2.30×                     |

¹ n_prompt=131000. bf16 column is the A/B control measured in the SAME session on the SAME binary
(the `PLOW_NV_W8A8=1` cubin runs a bf16 pkt via the untouched `GEMM` path).

**Verdict: w8a8 is the FIRST prefill lever in the campaign to BEAT plow bf16** (−5…−16.5%), exactly
where T6 w8a16 (dequant-to-bf16-in-smem) was NEGATIVE (+0.4…+3.5%) — true fp8 tensor cores deliver
where halving weight bytes did not, confirming prefill GEMM is **compute-bound**. **But plow w8a8
does NOT beat vLLM fp8 TTFT**: it is 1.56–2.29× behind, ≈ the same relative gap as plow bf16 vs
vLLM bf16 (1.44–2.30×), because both engines gain similarly from fp8 (vLLM fp8 ≈ −24% vs its bf16;
plow w8a8 −5…−16.5% vs its bf16). vLLM's paged FA-3 + fused fp8 GEMMs keep the lead; the remaining
gap is FLASH + per-op launch/counter overhead, NOT the GEMM.

**Per-op trace (block-0, one 16k chunk, `PLOW_NV_TRACE=1`), body-cycle share:**

| op | bf16 | w8a8 |
|----|------|------|
| GEMM / GEMM_FP8            | 39.4% | 37.5% |
| GEMM_GLU / GEMM_GLU_FP8    | 41.4% | 33.0% |
| QUANT_FP8                  |   —   |  6.2% |
| FLASH_PREFILL              | 16.1% | 19.6% |
| GEMM-family total          | 81.1% | 70.7% |

The 2×-rate mma cuts GEMM **body cycles ~28%** (not 2×): the tile stays 128×128 and the cp.async
staging + pipeline + epilogue overhead is unchanged, and the per-row QUANT_FP8 adds 6.2%. Net
−17.4% total body cyc ⇒ −5…−16.5% wall. GEMM share drops 81%→71%; **FLASH_PREFILL is the new
co-bottleneck** (16%→20%) and grows with ctx — which is why the win shrinks −16.5%@4k → −5%@128k
(flash/HBM dominates long-ctx and w8a8 does not touch it).

**Gates:** 1 (dep-graph, CPU blob walk) — all 280 GEMM_FP8 wait on their QuantFp8, topo-ordered;
192 quants each feed ≥1 GEMM; `PLOW_UNISEG=1` ⇒ all stream entries coarse. 2 (oracle,
`PLOW_NV_W8A8=1`) — quant_fp8 relL2=0 ×2, gemm_w8a8 0…5.2e-5 ×4, gemm_glu_w8a8 5.3e-5…9.6e-5 ×2 ALL
PASS; w8a8-vs-full-precision relL2 3.6% (standard e4m3). 3 (e2e correctness) — plow BF16 output is
TOKEN-IDENTICAL to vLLM bf16 on all 5 prompts (baseline validated); w8a8 first-token AGREES 4/5
(win5=1156-tok window-crossing flips). **Divergence budget** (token-match depth, w8a8-vs-own-bf16 |
vLLM-fp8-vs-own-bf16): short 1|1, france 48|6 (identical where overlapping), poem 38|5, win5 0|0,
p4k 32|**0** — **plow w8a8 drifts from its bf16 NO MORE than vLLM fp8 drifts from its bf16, and LESS
on poem/p4k**; win5 flips the first token for BOTH engines (near-tie window-crossing prompt). 4 —
above. Decode/Qwen cubins byte-identical; ptxas 238 regs / 0 spill / occ-1 UNCHANGED.

## T5-cp.async-KV-stream (2026-07-19) — the flash L1 lever (rtx-07 campaign)

`d_flash_prefill` (op_attention.cuh, `PLOW_NV_FA_PIPE=1`, now the header default) stages the
FLASH_PREFILL K/V tiles through the same cp.async ring discipline T3 proved for the tiled GEMM.
The T4 kernel staged each tile with plain vectorized loads + `__syncthreads`, so the mma stalled
on the whole tile landing from HBM (the 128k HBM-KV-bound tail).

- **Fix:** stage K **NATURAL** `Ks[kv][hd]` (was `KsT[hd][kv]` transposed — a scatter cp.async
  cannot express) and read the QK^T B fragment with `ldmatrix.x2` **non-`.trans`** (the T3-proven
  bit-exact equivalent), so both K and V are contiguous `[kv][hd]` and fill smem via `cp.async.cg`
  16-byte lines. A full cross-tile double-buffer of BOTH operands (+33 KiB) exceeds the **99 KiB
  sm_120 opt-in cap**, so the pipeline stays SINGLE-buffered and still overlaps both loads by
  exploiting operand lifetimes: **K is dead after QK** so `K[t+1]` is prefetched right after `QK[t]`
  (overlaps `softmax[t]`+`P·V[t]`); **V is needed last** so `V[t]` loads at the tile top (overlaps
  `QK[t]`+`softmax[t]`); a last-tile empty `commit_group` keeps the `wait_group<1>` count uniform.
- **Registers/occupancy:** 238 regs (was 236), 0 spill, occ-1. Dynamic smem SHRINKS 83.25→**79.75
  KiB** (natural K is smaller than the transposed `KsT`), so occupancy is unchanged. Decode gemma
  object (150 regs) and default Qwen object (155 regs) cubins **BYTE-IDENTICAL** across the flip.
- **Bit-exactness:** GPU-verified **LOGITS BIT-IDENTICAL** base vs pipe (p_short) — same mma bytes
  in the same lanes ⇒ f32 accumulation identical ⇒ greedy tokens unchanged everywhere.

**Gates:** A — flash-prefill 40/40 oracle PASS (relL2 1.5e-3…2.4e-3, unchanged) incl. 5 new
pipeline-edge cases (ragged last tile, mid-tile window floor, hd512 straddle+ragged, chunk ns3);
GEMM/fp8 unchanged; wave64 negctrl still FAILS every flash case. B — prefill==decode-only first
token (p_short 236761), window-crossing p5_win 24-token continuation IDENTICAL, 32k/64k/128k
device==host argmax AGREE + finite. D — decode/Qwen cubins byte-identical.

**Next lever:** the remaining 128k 2.29× is now GEMM-family + fixed per-launch overhead + the
un-double-buffered flash (the 99 KiB cap blocks a true 2-stage ring). L2 (occupancy bucket-split)
is a MEASURED NEGATIVE — see below.

## T5 L2 — occupancy bucket-split: NEGATIVE (blocked at three levels)

The GEMM phases run occ-1 in the combined `_pf` megakernel. Splitting prefill into an occ-2
GEMM-family bucket + occ-1 flash bucket was studied against the emitter + persistent model and is
**not viable**; recorded as a negative result:

1. **Occ-2 needs ≤50 KiB block smem; no prefill object qualifies.** sm_120 caps opt-in smem at
   99 KiB/block and 100 KiB/SM, so 2 blocks/SM ⇒ ≤50 KiB each. The flash arena is 79.75 KiB
   (2× = 159 KiB) and even a GEMM-ONLY object is 60 KiB after T3's 3-stage ring (2× = 120 KiB) —
   both exceed 100 KiB/SM. Occupancy here is **smem-bound, not register-bound**, so the cheap
   `__launch_bounds__` knob buys nothing (ptxas already fits 238 regs at occ-1; more warps can't
   be resident). Getting a GEMM object to ≤50 KiB means dropping the cp.async depth T3 proved is
   the −72% win — a net loss.
2. **The packet structure forbids a clean split.** Each of the 6 prefill bucket programs
   (`emit_phase`, Mode::Prefill) is ONE cooperative launch containing all 48 layers interleaved
   per-layer: qkv-GEMM → FLASH → o-GEMM → gate|up-GEMM_GLU → down-GEMM, with FLASH[L] depending on
   qkv-GEMM[L] and o-GEMM[L] on FLASH[L]. GEMM and flash cannot be hoisted into two sequential
   phases without materializing every layer's intermediates.
3. **Two objects ⇒ two grids ⇒ the megakernel premise dies.** The counter protocol is per-grid;
   two cooperative launches can't share counters, so a GEMM-object/flash-object split needs host
   `cudaDeviceSynchronize` at every sublayer boundary (~96 grid syncs/chunk) — exactly the
   per-op-launch model the persistent interpreter exists to avoid.

**Serve-side design note (cubin naming, for the serve agent who owns `crates/plowrt`):** the T5
prefill kernel is BIT-IDENTICAL to T4, so `interp_sm120_pf.cubin` can adopt it transparently — no
new cubin name, no loader change, no correctness gate beyond the byte-identity already proven. If
L2 were ever pursued it would need a SECOND prefill cubin (e.g. `interp_sm120_pf_gemm.cubin`) plus
a per-phase launch loop in `exec/gpu.rs`, but per the negative result above that trade is a loss on
this hardware and is not recommended.

**T4 delta vs T3:** −27.9% @4k (1.39x), −38.3% @16k (1.62x), −43.3% @32k (1.76x),
−43.1% @64k (1.76x), −31.7% @128k (1.47x). The win PEAKS at mid-context (32–64k) rather
than the flat 1.5–2.2x band projected: at 4k the single 4096 chunk has a smaller O(ctx²)
flash share so the P·V lever buys less; at 128k the tail flash P·V is HBM-KV-read-bound
(mma gain falls to 2.29x there) and 40/48 layers are sliding O(window). The isolated
flash-op A/B is 2.3–4.3x (`gemma4-12b-plow-flash-pv-sm120.json`); the end-to-end dilution
is exactly the non-flash GEMM share (40.9%) that T3 already fast-pathed.

## T4-mma-pv-prefill (2026-07-19) — MERGED rtx HEAD (T3 a026a72 + T4 af4a953)

Register-resident `mma.sync m16n8k16` P·V (HD-split warp partition) retires the FFMA-serial
online-softmax P·V that T3 left as the dominant prefill cost (57.7% of the 8192-chunk). T4 also
fixes a pre-existing sliding-window tile-skip bug (the per-tile skip used `qabs_max` = the newest
query, dropping tiles the older queries still attend at the window's trailing edge) that corrupted
every sliding layer (40 of 48) and was masked by a degenerate oracle reference.

- **Binary:** `gemma4_sm120_chat` rebuilt from committed HEAD (`101625614…`, merge of a026a72 + af4a953,
  both verified in history). Single packet `gemma4-12b-c128k.pkt` (max_ctx=131072, PLOW_UNISEG=1).
- **ptxas -v (merged HEAD source):** prefill `_pf` object 236 regs / 0 spill / occ-1; decode gemma
  object 150 regs, default Qwen 155 regs — both unchanged (prefill-only op-body edits). Decode SASS
  byte-identical: the bf16 France decode-only token stream is identical between the pre-merge T3
  binary and the HEAD build.
- **Parity (GATE B):** prefill == decode-only first token — short100 236761=236761, 4k 236779=236779
  (both AGREE, match T3). Window-crossing p5_win (1156 tok > 1024 window): prefill and decode-only
  give the IDENTICAL 24-token greedy continuation (stable across the sliding fix). 32k prefill
  finite/sane (top-1 +10.375, +1.6 margin, device==host argmax, non-degenerate continuation).
- **bf16 decode spot-check @4k:** 18.58 ms/tok (Phase-0 18.30 / G7-bf16 18.33) — unchanged within
  run variance, as expected (decode object byte-identical through T3/T4).

**Bottom line (bf16 prefill):** T4 closes every rung vs both T3 and vLLM (12.11x→ from 17.7x at
128k, 1.89x→ from 2.6x at 4k). Still 1.9–12x behind vLLM's tuned paged FA-3; the remaining lever
is a cp.async KV-stream pipeline for the long-ctx flash P·V tail (the 128k HBM-bound regime).

## T3-gemm-pipeline (2026-07-19)

REAL cp.async pipeline for the prefill tiled GEMM (`d_gemm`/`d_gemm_glu`, `op_gemm.cuh`).
T2 proved the staging was synchronous (`pgm_stage_b` did plain loads; the cp.async
commit/wait ran on EMPTY groups). The blocker was the B (weight) operand stored TRANSPOSED
into smem (`[k][n]` scatter — 8 elems to 8 rows — which cp.async cannot express).

**Fix:** stage BOTH operands in their natural, K-contiguous layout (A `[m][k]`, B `[n][k]`)
via `cp.async.cg` 16-byte lines (the `src-size` operand zero-fills out-of-range M/N/K lines),
and read the mma B fragment with `ldmatrix.x2` **non-`.trans`** instead of the in-tree
`.trans` idiom. The non-trans B path was proven **bit-identical** to the old `[k][n]+.trans`
path by a standalone probe (`runtime/nvidia/experiments/t3_pipe_probe.cu`: 0/2048 register
mismatches, matches f32 CPU ref). Pipeline: plain GEMM rings **3** stages of (A,B); GEMM_GLU
rings **2** stages of (A,Bg,Bu) — `commit_group`/`wait_group<STAGES-1>` discipline. The mma
operands are the same bf16 values in the same lanes ⇒ f32 accumulation is bit-exact vs T2 ⇒
greedy tokens unchanged.

- **Registers/occupancy:** prefill object 236 regs (was 238), 0 spill, occ-1. Dynamic smem
  UNION unchanged at 77.5 KiB (the GEMM arena grew 37.0→60.0 KiB but stays under flash's
  77.5 KiB claim, so the megakernel schedule is byte-for-byte T2's). Decode gemma object
  SASS **byte-identical** to T2.
- **Effective throughput @4k:** 98 TFLOP (2·N·P proxy) / 0.849 s = **115.4 TFLOP/s** end-to-end
  (~46% of the 251.9 bf16 peak), up 3.55× from T2's 32.5. This whole-prefill proxy now
  under-states the GEMM kernel itself (GEMM is only ~41% of the traced chunk).
- **Per-op (block-0 body cycles, first 8192-chunk of 16k):** FLASH_PREFILL 57.7%,
  GEMM_GLU 20.7%, GEMM 20.1%, HEADNORM_ROPE 0.65%, else <0.6%. GEMM family 83.7% (T2) → 40.9%.

**Gates:** A — GEMM/GEMM_GLU oracle PASS (relL2 1.3e-6…1.5e-4, ~1e-4 band), full suite ok,
wave64 negctrl FAILS. B — prefill == decode-only first token (short100 236761, 4k 236779) and
IDENTICAL to the T2 baseline binary on the same prompts. D — decode object SASS byte-identical.

**Next levers:** (a) register-resident mma.sync flash P·V (hd256) — now the biggest single
cost; (b) per-opcode GEMM tile variants (small-M lm_head / short buckets); (c) T5 occupancy
bucket-split (emitter change, out of scope).
