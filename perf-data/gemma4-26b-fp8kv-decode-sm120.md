# Gemma-4-26B-A4B — plow fp8-KV decode vs vLLM — sm_120 (2026-07-22)

First measurement of **plow fp8 weights + fp8 (e4m3) KV cache** (`PLOW_FP8_KV`) on the
26B-A4B MoE. rtx-19 **E3 was gated on 12B/31B only** (both dense); the P9 26B campaign
recorded fp8-KV as *"compiles but decode TRAPS ~step 1 on 26B geometry (hd256 kvh8 rings +
hd512 k_eq_v full); 31B-only feature so far."* This run establishes that the trap is **GONE**
on current main (fixed by the concurrent oracle-review hardening `3bd8874`, whose only fp8-KV
change wraps null-ptr scale arithmetic in `if constexpr (FP8KV)` — inert for the FP8KV=true
decode path), and quantifies the fp8-KV decode ladder against the vLLM fp8kv baseline.

One RTX PRO 6000 Blackwell (sm_120, 188 SM, CUDA 13.0), TP1, batch 1. Harness
`gemma4_sm120_chat`, `PLOW_PREFILL=0` decode-only KV warm, 16 warmup + 120 timed steps,
seed-0 vLLM RandomDataset prompt p0 per ctx (within-run sd < 0.02 ms, < 0.3% of mean).
vLLM reference: `gemma4-26b-a4b-vllm-sm120.md` (trusted baseline, not re-derived).

## Config

- **Measured at commit `7fd19fb`** (a same-day main commit). The fp8-KV trap fix relative to the
  P9 report is `3bd8874` (oracle-review; its fp8-KV change wraps null-ptr scale arithmetic in
  `if constexpr(FP8KV)`, touching only the bf16 FP8KV=false instantiation — inert for the
  FP8KV=true path measured here). Main has since advanced past `7fd19fb` with **perf-relevant
  interp changes not re-measured here** — `4310a06` (workgroup successor-counter signaling
  fan-out) and `08896b4` (devinst64 packet layout). This commit is rebased onto the current tip
  for cleanliness, but the absolute TPOT below is the `7fd19fb` build; current-main absolute
  numbers may differ by ~0.1–0.2 ms (uniform interp tuning, applies to all plow configs). The
  **verdict is robust to that delta** — the fp8-KV win in the mid-range and the ~1.8× long-ctx
  KV-read slope (which leaves 96k/128k to vLLM) are structural, not a signaling tweak.
- Binary: `cmake -B build-gf4-fp8kv -S runtime -DPLOW_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_FLAGS="-DPLOW_NV_FA_GF_FULL=4" -DPLOW_NV_W8A8=ON -DPLOW_FP8_KV=ON` (plain env).
- Packet: `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=48 PLOW_FP8=1 PLOW_FP8_HEAD=1 PLOW_FP8_KV=1 gemma4
  <dir> 132096 out.pkt 188` (601 packets, KV cache 2.86 GiB vs 5.64 GiB bf16-KV = −49%).
- fp8 twins reused: `/workspace/models/gemma-4-26B-A4B-it/fp8-full-plow` (2 shards, incl. head).
- Default byte-identity: bf16 packet (my flags off) md5 `04c807bd2d5c862e446406b7bc2bcdb8` ==
  committed `gpu-assets-26b/b1/gemma4-26b-ctx132096-b1.pkt` (emitter untouched on this branch).
  fp8-KV decode object: **222 regs, STACK 0** (no spill).

## Decode TPOT ms/token (batch 1; bold = plow fp8kv beats vLLM fp8kv)

| ctx  | plow fp8kv | plow fp8+head¹ | plow bf16¹ | vLLM bf16 | vLLM fp8 | vLLM fp8kv | fp8kv vs fp8kv |
|------|-----------|----------------|-----------|-----------|----------|------------|----------------|
| 1k   | 6.126 | 5.84 | 8.24 | 7.61 | 5.76 | 5.92 | +0.21 (loss) |
| 4k   | **6.149** | 5.89 | 8.29 | 7.90 | 6.08 | 6.19 | −0.04 |
| 16k  | **6.437** | 6.20 | 8.61 | 8.64 | 6.82 | 6.62 | −0.18 |
| 32k  | **6.702** | 6.67 | 9.07 | 9.57 | 7.74 | 7.28 | −0.58 |
| 64k  | **7.442** | 7.54 | 9.94 | 10.33 | 8.63 | 7.52 | −0.08 |
| 96k  | 8.118 | 8.40 | 10.81 | 11.34 | 9.54 | 7.94 | +0.18 (loss) |
| 128k | 8.791 | 9.16 | 11.57 | 12.34 | 10.48 | 8.46 | +0.33 (loss) |

¹ plow fp8+head / bf16 rows are the committed P9 3-prompt means (`gemma4-26b-plow-sm120.md`);
the fp8kv column is this run's p0 (methodology offset ≈ +0.1 ms vs a 3-prompt mean, seen at 1k
where p0 fp8+head reproduced 5.94 vs committed 5.84). Read with that offset: **16k (−0.18) and
32k (−0.58) are clear wins; 96k (+0.18) and 128k (+0.33) clear losses; 1k (+0.21) a loss. 4k
(−0.04) and 64k (−0.08) are within the offset = marginal wins / effective ties.** The overall
shape (fp8-KV wins the mid-range, loses long-ctx) is robust to the offset.

### What fp8-KV moved (fp8kv vs plow fp8+head, same fp8 weights)

| ctx  | fp8+head | fp8kv | Δ | 
|------|----------|-------|-----|
| 1k–16k | 5.84–6.20 | 6.13–6.44 | **+0.24…+0.29 (SLOWER)** — dequant ALU > KV bw saved |
| 32k  | 6.67 | 6.70 | +0.03 (tie) — the crossover |
| 64k  | 7.54 | 7.44 | −0.10 (FASTER) |
| 96k  | 8.40 | 8.12 | −0.28 (FASTER) |
| 128k | 9.16 | 8.79 | −0.37 (FASTER) |

Crossover at ~32–64k — same shape as E3's 31B result (crossover by 32k). fp8-KV is the faster
plow fp8 config from 64k up; fp8+head is faster below.

## Verdict — sub-goal 1: PARTIAL (fp8-KV unblocked on 26B, extends lead to 64k; 96k/128k NOT beaten)

- **fp8-KV now works on 26B** (previously trapped). All 7 ctx: argmax device==host AGREE, sd
  < 0.02 ms. Trap resolved; this is the first valid 26B fp8-KV decode measurement.
- **Beats vLLM fp8kv ("the config to beat ≥16k") at 4k, 16k, 32k, 64k** — including flipping
  the 64k point plow fp8+head only *tied* (7.54→7.44) and the 96k/128k deficit **halved**
  (fp8+head was +0.46/+0.70 → fp8kv +0.18/+0.33).
- **Does NOT reach the target of beating vLLM fp8kv at 96k (7.94) / 128k (8.46).** Cause is an
  attention-kernel slope, not weights: plow's marginal KV cost 32k→128k is ~21.3 ns/tok vs
  vLLM's ~12.0 ns/tok (~1.8×). Weights are flat with ctx, so the entire long-ctx deficit is
  the fp8 `FlashDecodeFp8` hd512 full-attn read (the 5 full layers; sliding layers ring-cap at
  the 1024 window). Per-element e4m3→bf16 dequant per split + split/merge overhead is the
  steeper slope. Closing it is a real flash-decode kernel task (grid-aligned split like T9b's
  bf16 −3.4%, dequant-once-per-row, wider vectorized load) — out of scope for build+measure.
- **Best-of-both plow fp8** (min of fp8+head, fp8kv per point): 5.84/5.89/6.20/6.67/7.44/
  8.12/8.79. Beats vLLM fp8kv at 1k–64k; loses 96k/128k. Still loses vLLM fp8 at 1k (+0.08).

## Correctness gate

- Oracle `sm120_interp_op_test`: **ok** — full suite PASS incl. w8a8 GEMM (relL2 0 / 5.9e-5).
- **Real-text parity (E3-class), 3587 natural tokens, fp8-KV vs bf16-KV, same fp8+head weights:**
  first token **MATCH** (236865), **20 identical greedy tokens**, step-0 logit **relL2 4.96%**,
  top-2 identical. Squarely vLLM-fp8kv-class (E3 band 2.7–5.9%); NOT bit-exact (e4m3 KV, expected).
- Note: on a **random-token** vLLM RandomDataset prompt the step-0 relL2 is 68.9% and first
  tokens differ — a **degenerate-input artifact** (same class as E3's 82% periodic case; both
  configs just loop on garbage input). Real text is the valid parity signal, per E3.
- Token-identity within each fp8kv run: device/host argmax AGREE at every ladder point.

## Sub-goals 2 & 3 (this campaign) — NO-GO, evidence-based

- **Sub-goal 3 (fp8 1k gap 0.08 ms vs vLLM fp8): NO-GO, no GPU spent.** E6 (decode op overlap)
  is committed as an *honest negative* — GEMVs are 95% of the step and BW-saturate all 188 SMs
  (no compute-idle to hide behind); the only independent op group (3 RoPE siblings) is already
  overlapped by the global-queue scheduler; forcing serial regresses +1.4–1.6%. E7
  (`PLOW_FUSE_ARGMAX`) is byte-identical/~0-perf **and gated `!fp8_head`** — inert for the
  fp8+head config that owns the 1k point. Neither lever can cheaply close 0.08 ms.
- **Sub-goal 2 (bf16 1k/4k, +0.63/+0.39 ms): NO-GO.** (A fresh ctx-1k decode trace was attempted
  with `-DPLOW_NV_TRACE_DECODE=ON`; the harness block-0 dump returned empty, so the committed T9a
  ctx-4k block-0 trace stands as the evidence.) That trace already localizes the gap at the worst
  point: not MoE — dense GEMV bodies (already at 60–84% of the 1535 GB/s ceiling,
  M=1 bandwidth-bound) + ~21.6% interpreter gate-wait on GEMV_QKV/GEMV. At M=1 a GEMV cannot go
  faster per byte (tensor-core/tiling is a batch lever); the only bf16 short-ctx levers are
  fewer bytes (SplitZip lossless, projected 8.24→7.0–7.2 but killed for multi-user) or cutting
  the interpreter intercept/gate (T9c scheduler surface). No cheap lever moves it.
