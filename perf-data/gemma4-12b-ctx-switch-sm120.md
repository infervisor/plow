# Gemma-4-12B fp8 DECODE — ctx-regime kernel switching (GF2↔GF8) + o_proj/down GEMV review

**Campaign beat12b-ctx-switch** · branch `beat12b-ctx-switch` (child of `beat12b-fp8-margin`) · **2026-07-23**
GPU: 1× RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, 96 GB). CUDA 13.0.

Goal: capture GF8's long-ctx flash win (campaign left it on the table: −0.73 ms @128k, blocked by
its +0.37 ms short-ctx register tax) **without** the short-ctx tax, by switching the decode kernel
by context-length regime — the sm_120 analogue of AMD's segment-based wave-size switching.

**RESULT: ctx-regime switching (GF2 <65536, GF8 ≥65536) DOMINATES fixed-GF2 at every rung —
statistical tie at 1k–32k, −1.1 %/−2.3 %/−5.3 % at 64k/98k/128k. Per-step switch cost is 0
(measured). Token streams identical to fixed-GF2 on natural text; switch mechanism byte-exact
pre-boundary. Shipped OFF by default (byte-identical).**

## HEADLINE — decode TPOT ms/token (mean, n=112, method of record = gemma4_sm120_chat)

Shipped fp8 config (fp8 weights + E5 fp8 lm_head + fp8-KV + the margin campaign's nsplit levers).
GF2 = shipped decode object (209 regs, ns47 full / ns16 sliding). GF8 = the `_gf8` decode twin
(GF_FULL=8, 234 regs, ns94 full). Switched = `PLOW_GF_SWITCH=65536` picks the object+program by
kvlen each step. argmax device==host AGREE at every point, all three configs.

| ctx | GF2 shipped | GF8/ns94 | **switched(65536)** | GF8−GF2 | **switched−GF2** | winner |
|------:|-----------:|---------:|--------------------:|--------:|-----------------:|:------:|
| 1024 | 10.927 | 11.311 | **10.929** | +0.384 | **+0.002** | tie |
| 4096 | 10.986 | 11.339 | **10.979** | +0.353 | **−0.007** | tie |
| 16384 | 11.204 | 11.410 | **11.211** | +0.206 | **+0.007** | tie |
| 32768 | 11.523 | 11.692 | **11.529** | +0.169 | **+0.006** | tie |
| 65536 | 12.199 | 12.060 | **12.068** | −0.139 | **−0.131 (−1.1 %)** | GF8 |
| 98304 | 12.882 | 12.596 | **12.591** | −0.286 | **−0.291 (−2.3 %)** | GF8 |
| 131072 | 13.685 | 12.960 | **12.964** | −0.725 | **−0.721 (−5.3 %)** | GF8 |

sd ≤ 0.026 ms at every point (contention-free lease). The 1k–32k ties are all within ±0.007 ms
(< ½ sd): the switched run takes the byte-identical shipped GF2 path there (short_steps=128,
long_steps=0). At 64k+, kvlen ≥ 65536 for the whole generation → GF8 (long_steps=128). **The
switched ladder is ≥ shipped-GF2 at every rung — acceptance met.**

## Threshold

GF8 crossover is in **(32768, 65536]**: GF8 loses at 32k (+0.169) and wins at 64k (−0.139).
Shipped threshold **65536** is conservative — GF8 engages only where it is *measured* to win, so
the switched ladder can never regress the shipped ladder. A 40–48k probe could lower it (linear
interp puts break-even ≈ 49k, worth ~0.05–0.10 ms at 48–64k); not pursued — the safe threshold
already captures the full 64k–128k win, which is the campaign's target band.

## Mechanism comparison (evaluated all three; A+C chosen)

### A. INTERPRETER RELAUNCH (per-step function switch) — CHOSEN, cost = 0

Decode is ONE cooperative launch per step (`plow_sm120_launch`, interp_sm120.cu). The flash work
count is computed **in-kernel** (`n_work = n_batch·(n_head/GF)·nsplit`, op_attention.cuh) — nsplit is
the only packet-side knob. "Relaunch" is therefore just choosing a different resident CUfunction per
launch. **Measured switch cost (alternate GF2/GF8 every step vs the mean of the two pure runs):**
1k 11.134 med vs mean-of-pure 11.111 → **+0.023 ms (0.2 %, within sd 0.19)**; 128k 13.318 vs 13.321
→ **−0.003 ms (0 %)**. The module is resident, the arena/grid are per-object (GF8 arena 24640 B vs
GF2 12352 B, both grid=188=occ-1), so picking the other function+program adds nothing. The
per-object arena difference is handled cleanly — each object's launch helper carries its own
compile-time arena, so no host bookkeeping.

### C. EMIT-SIDE two decode programs — USED (the nsplit half of the switch)

The two regimes need **different nsplit**: GF2 is grid-aligned at ns47 (376 items = 2/block on 188
SMs), GF8's n_grp=2 aligns only at ns94 (188 items = 1/block). A single shared packet fails both
ways (see NO-GO below), so switching carries a second decode **program** (ns94) alongside the second
**object** (GF8). The two packets are tensor-table-identical (cmp: only the decode flash op's nsplit
field + dep map differ; `opart`/`mlpart` are sized by the prefill buckets so ns94's decode partials
fit the ns47 packet's scratch) → the long program shares the shipped packet's weights+scratch, zero
extra VRAM. In the harness this is a second `PLOW_PKT_LONG` blob whose decode program binds the same
tensor table; the production form emits both decode programs into one blob (free selection, blob
grows by one ~small decode program).

### B. RUNSEG (multi-segment relaunch) — REJECTED for decode

RUNSEG (PLOW_NV_SEGMENTS, T9c; +0.44 % on prefill) relaunches once per **wave-class segment**
(contiguous op-id window) *within one program*, and CAN carry a different object per segment (T10
already runs GEMM segments on `_pfgemm` occ-2 and flash on `_pfseg` occ-1). But it segments **within
a step**; the ctx regime is a **between-step** property (kvlen). Using RUNSEG to put each layer's
flash on a GF8 object and its GEMVs on GF2 within every step needs ≈100 relaunches per decode step.
Decode is single-launch **latency-bound** (~11 ms, host prologue 0.04 ms + one sync); the K−1
relaunch+sync tax that cost +0.44 % on compute-heavy prefill would cost tens of percent on decode.
Mechanism A reaches the same GF-per-regime goal in one launch at zero cost. Read
gemma4_sm120_chat.cu:760-800 + the +0.44 % T9c number — rejected.

## Correctness gates

- **Switch mechanism is byte-exact.** Cross-threshold run at ctx=16384, threshold=16392: the 7
  pre-boundary decode tokens (+ prefill token) are byte-identical to fixed-GF2. The switch itself
  introduces zero error.
- **Token identity on natural text: PASS.** 214-token natural-prose prompt, 64 generated tokens,
  threshold=230 (crosses at decode step 16, 15 GF2 + 49 GF8 steps): fixed-GF2, fixed-GF8, and
  switched streams are **all 64/64 tokens byte-identical** (diff clean). GF8 is numerically
  equivalent to GF2; greedy is stable on real text.
- **GF8 near-tie on high-entropy prompts (documented class).** On the 128k-random-token prompt
  (adversarial high-entropy logits) GF8-from-start greedy-flips within a few steps — the SAME class
  the margin campaign characterized for GF8 step-0 logits (byte-identical at 1k/4k; random-prompt
  flips) and for the shipped fp8-KV drift. Not a correctness fault: reduction-order rounding on
  near-ties. The switched run tracks fixed-GF2 exactly to the boundary, then continues coherently.
- **Flash output bit-identical.** flashdec_fp8_bw_12b microbench: GF8/ns94 `maxdiff_vs_ref = 0.0000`
  vs the GF2 reference at every ctx (1k…128k).
- **argmax device==host AGREE** at all 7 ladder points × {GF2, GF8, switched}.
- **sm120_interp_op_test: ok** on the shipped GF2 object (shared op bodies; the GF8 twin differs
  ONLY in the flash template's GF parameter, validated by the microbench + argmax above — the op
  test links the Qwen3 default object whose gqa=4 cannot instantiate GF8).
- **Default-off = byte-identical.** `PLOW_GF_SWITCH` unset → the harness takes the exact shipped
  GF2 path; `PLOW_NV_GF8_TWIN` default 0 → the decode object is byte-identical; the ns94 packet is
  produced only via `PLOW_NS_FULL_ABS=94` (never the default).

## WORK ITEM 2 — o_proj / down fp8 GEMV bandwidth review

Single-block microbench `gemv_fp8_bw_12b.cu`, grid = n_cu = 188, M=1 arena path (the real decode
GEMV), min-of-300, **L2-flushed cold HBM read** (each weight is read once per token in-network → a
cold stream; a naive L2-hot probe reports 1400–2850 GB/s and is meaningless here). Real 12B dims
confirmed from the emitter/config — **the brief's "down K=9728" is Qwen3-4B's intermediate; 12B down
is K=15360, o_proj K=4096.**

| arm | N | K | rows/block | cold GB/s | uint4 | balanced-N |
|-----|--:|--:|-----------:|----------:|------:|-----------:|
| lm_head | 262144 | 3840 | 1395 | **1355** | 1360 | 1352 |
| gate/up | 15360 | 3840 | 82 | **819** | 811 | 800 |
| down | 3840 | 15360 | 21 | **633** | 638 | 638 |
| q_proj | 4096 | 3840 | 22 | **549** | 530 | 530 |
| o_proj | 3840 | 4096 | 21 | **512** | 525 | 522 |

(Reproduces the margin campaign's in-network cold numbers: gate/up 819≈823, down 633≈640.)

**Cause — named, with numbers: small-N SM-fill / latency-hiding wall, not the kernel.** Cold GB/s is
monotone in rows/block (= N/188 = how deeply each SM is filled): 1395→1355, 82→819, 21→512-633. The
weight loads are already perfectly coalesced (256 B/warp-pass, contiguous) and the dequant is the
hardware fp8x2→f32 cvt (not the bottleneck: L2-hot the SAME kernel hits 1355-2850 GB/s). o_proj and
down write only **N=hidden=3840 → 20-21 rows/block → ~2.5 rows/warp** at occupancy-1 (the 209-reg
megakernel): too few concurrent weight-row reads in flight to hide the ~500 ns HBM latency on a cold
stream. gate/up write N=15360 (82 rows/block) → deep pipeline → 819; lm_head N=262144 → 1355 (near
the practical ceiling). o_proj is worst because it *also* has the shortest K (4096 → 2 unroll groups
of loads per row).

**Verdict: at the small-N floor for this occ-1 megakernel. NO cheap fix.**
- **uint4 (16 B/lane) loads — NO-GO.** Cold-read A/B: o_proj +2.5 %, down +0.8 %, q_proj −3.5 %,
  gate/up −1.0 %, lm_head flat — all within noise, net wash, one regression. The arm is
  HBM-latency-bound (waiting on the cold read), not load-issue-bound, so halving the load
  instructions buys nothing. Disproves the "load-width (uint4)" idea the margin campaign left open.
  (The flash arm's PLOW_FP8_FAST uint4 win does NOT transfer: flash was dequant-ALU-bound, this is
  latency-bound.)
- **Balanced N-partition — NO-GO.** Activating the 5 idle tail blocks (per=ceil(N/188) idles slices
  183-187 for N=3840): o_proj +2 %, others flat/worse — within noise, not worth the code change.
- **The only real lever is split-K decode GEMV** (partition the K reduction across more blocks + a
  merge) to raise memory-level parallelism on small-N shapes — but that is a new op with a reduction
  cost, not a cheap fix, and decode is occ-1 (can't add occupancy the simple way). Roofline: o_proj
  512 / gate/up-class deep-fill ceiling ~819 = the shape is fill-limited, and the *aggregate* GEMV
  already runs ~90 % of the fp8-byte ceiling once weighted by achievable-per-shape. Honest floor.
- A/B'd against the full 5-shape suite; no regression-free win. **Ship nothing.**

## Files / reproduce

- Twin object: `runtime/nvidia/interp_sm120.cu` (`PLOW_NV_GF8_TWIN` → `_gf8` symbols);
  `runtime/CMakeLists.txt` (`plow_interp_sm120_gemma_gf8`, GF_FULL=8).
- Switch harness: `runtime/tests/gemma4_sm120_chat.cu` (`PLOW_GF_SWITCH=<kvlen|alt|off>` +
  `PLOW_PKT_LONG`; default off = shipped path).
- Microbenches: `runtime/tests/flashdec_fp8_bw_12b.cu` (+GF2/ns94 row, +98304),
  `runtime/tests/gemv_fp8_bw_12b.cu` (cold-read arg `<iters> <flush>`).
- ns94 packet: `PLOW_NS_FULL_ABS=94 PLOW_UNISEG=1 PLOW_FP8=1 PLOW_FP8_HEAD=1 PLOW_FP8_KV=1 gemma4 …`.
- Switched ladder point: `PLOW_GF_SWITCH=65536 PLOW_PKT_LONG=g12b_ns94.pkt PLOW_FP8_DIR=… \
  gemma4_sm120_chat g12b.pkt <model> ids_<ctx>_p0.bin 128`.

## plowrt production wiring (host-side policy, the shipping home)

The chat harness is the method of record and implements the full mechanism (measured above). The
plowrt change mirrors it exactly and is mechanical: `GpuEngine::load` (exec/gpu.rs) loads a second
module from `PLOW_NV_CUBIN_LONG` (or a second blob cubin section), reads its `plow_arena_bytes`, and
the decode launch picks the CUfunction + decode program by the slot's kvlen when
`PLOW_GF_SWITCH=<ctx|auto>` is set (default off = current byte-identical path). With bounded
multi-step (PLOW_MULTISTEP=K) the switch granularity is the quantum — the threshold need not be
exact (crossover band is wide). Not wired into plowrt in this pass; the mechanism, cost, and
correctness are all proven in the harness.

## NO-GOs / honest notes

- **One shared ns94 packet for both objects — NO-GO.** GF2/ns94 taxes short ctx: flash+merge @1k
  0.0302 vs the aligned ns47 0.0209 (+44 %), @4k +50 % (752 items = 4/block + doubled merge partials
  vs the aligned 376 = 2/block). And GF8/ns47 gives 94 half-filled items → 0.44 ms @128k regression.
  The regimes need different nsplit → switching must switch the program, not just the function.
- **Threshold not tightened below 65536** — 40–48k unmeasured; safe 65536 already captures the
  target band.
- **GF8 register tax is real** (+0.37 ms @1k, ctx-independent) — that is exactly why switching, not
  a global GF8 rebuild, is the answer; the tax is never paid below the threshold.
