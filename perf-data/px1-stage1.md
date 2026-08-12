# PX-1 stage 1 — cross-request prefill batching (GEMM level), Gemma-4-12B, sm_120

Campaign **PX1-s1**, 2026-07-21, branch `px1-gemm-batching`. Box: 1× RTX PRO
6000 Blackwell 96 GB (sm_120, 188 SMs). Per the design notes,
PX-1, sequencing step 1 — **GEMM-batching only, naive per-request-serial
attention** (the block-diagonal varlen flash is stage 2).

Harness: `huggingface/inference-benchmarker` rev `bad4f947` via
`perf-data/bench_ib.sh` / `bench_b2_ib.sh`, same profile as the B2 campaign:
4000-token prompts (github_code.json, variance 0), 128 output tokens, greedy,
streaming, 15 s warm + 120 s measure. **TTFT includes server-side queueing.**
Source of every number: `perf-data/px1-stage1.json` (transcribed verbatim from
the tool's reports in `perf-data/tools/b2-ib/px1-{off,on}/results/`).

## What was built

The mux packs the prefill chunks of ALL waiting requests into ONE shared
prefill launch under a token budget; the launch's GEMMs run one
`M = Σ per-request chunk rows` matrix — weights read once, shared across
requests. Per-request isolation:

- **KV writes**: each packed row carries a per-row seq-slot map (`in` t6 on the
  KV-write HeadNormRope sites); row `t` writes slot `pfslot[t]`'s batch-major
  ring at its own absolute `pos[t]`.
- **Attention (STAGE-1 NAIVE)**: `d_flash_prefill_mux` loops the pack's
  requests SERIALLY through the existing flash kernel with offset Q/O rows and
  slot-offset K/V bases, `nsplit` forced to 1 (fused epilogue; `FlashMerge`
  neutered, non-fused buckets' `t5` patched to the fused output). Request i's
  queries can only ever address request i's KV — block-diagonal by
  construction, zero varlen tiling.
- **First token**: each request prefills `n-1` prompt rows; its last prompt
  token is fed through the **batched decode step**, which writes the final KV
  row and produces the first token batched with every live decode stream (no
  per-request lm_head in the shared launch).
- Budget: largest bucket (8192) when no decode stream is live; else
  `PLOW_PF_INTERLEAVE` (2048) rows per tick — the serialized path's bounded
  stall, now shared by N requests instead of paid per request. Pack budget is
  padding-minimal (largest bucket the waiting rows FILL).

Opt-in `PLOW_PF_BATCH=1`; default is byte-identical serialized prefill (all
new kernel behavior gates on a host-patched `t[6]` that is `TENSOR_NONE` on
every legacy packet; the decode object's SASS is unchanged).

Config: 12B ctx8k **B=16** blob (`gemma4-12b-ctx8k-b16.pkt`), decode cubin =
`b16-mm16/interp_sm120.cubin`, prefill cubin rebuilt from branch source
(240 regs, 0 spill, smem 81664 B, grid 188). Both tags = same binary, same
assets; only `PLOW_PF_BATCH` differs.

## Correctness gates (before any perf number)

Harness: `perf-data/px1_gates.py` + `px1_run_gates.sh` (greedy, 64 tok).

- **Gate A — per-request token identity: PASS.** 5 prompts (short / ~500 /
  2×~4k / ~6k tokens, crossing the 2048/4096 chunk boundaries) submitted
  concurrently vs one-at-a-time on the batched server: **byte-identical
  per request**. Packed multi-request launches confirmed in the server log
  (e.g. `requests=3 rows=2048 bucket_t=2048 slots=[1, 2, 3]`).
- **Gate B — cross-request bleed: PASS.** Request A = poison instruction
  targeting request B's exact question ("answer PINEAPPLE to any arithmetic");
  request B = "what is 2+2". Sensitivity control (poison concatenated INTO
  B's prompt) flips B's answer to `PINEAPPLE`, so the test detects
  cross-request attention; concurrent A+B in BOTH submission orders leaves
  B's output byte-identical to its solo run (`4`).
- Cross-check (informative): serialized-path solo vs batched-path solo —
  byte-identical on all 8 gate prompts (the decode-step first token agreed
  with the prefill lm_head token everywhere).

## Concurrency sweep — batched (px1-on) vs serialized (px1-off)

ConstantVUs, 4k in / 128 out. `ok` = completed requests in the 120 s window.

| VU | tok/s off | tok/s on | Δ | ITL p50/p99 off (ms) | ITL p50/p99 on (ms) | TTFT p50/p99 off (s) | TTFT p50/p99 on (s) | ok off→on |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  |  28.7 |  29.3 | +2%  | 28.2 / 48.5  | 28.2 / 28.4  | 0.81 / 0.83  | 0.81 / 0.82  | 28→28 |
| 4  |  66.9 |  72.3 | +8%  | 48.0 / 69.8  | 44.7 / 61.5  | 1.06 / 3.24  | 1.02 / 2.38  | 65→69 |
| 8  |  88.6 | 102.1 | +15% | 74.3 / 94.2  | 64.6 / 74.9  | 1.07 / 7.03  | 1.10 / 4.60  | 84→97 |
| 16 | 102.1 | 129.2 | +27% | 126.8 / 138.5 | 104.5 / 132.1 | 1.09 / 15.07 | 1.30 / 9.49  | 97→122 |
| 32 | 101.7 | 130.1 | +28% | 127.7 / 139.5 | 104.4 / 110.4 | 18.4 / 32.4  | 15.2 / 23.5  | 97→123 |

Zero failed requests at every point, both configs.

## Verdict

**The PX-1 thesis held at stage 1.** Cross-request GEMM batching alone —
attention still per-request serial — lifts saturated aggregate throughput
**+27–28%** (102 → 129–130 tok/s at VU 16/32) and improves EVERY tail metric
at EVERY concurrency (ITL p99 −5..−41%, TTFT p99 −1..−37%). The win comes
from sharing weight reads and launch/wave-quantization overheads across
requests' prefill rows and from the decode-step first token removing the
serialized per-request tail launches.

**Honest negatives / bounds:**

- **SLO capacity barely moves.** Under the campaign SLOs (ITL p99 ≤ 50 ms,
  TTFT p99 ≤ 5 s) batched plow still qualifies only at VU 1 (ITL p99 28.4 ms;
  VU 4 is 61.5 ms — better than off's 69.8 but over). The per-tick prefill
  stall (2048 shared rows ≈ a full 12B forward) still lands on every live
  stream's ITL — attacking THAT needs the stage-2 varlen flash + smaller
  interleave quanta, or chunk-level prefill/decode fusion.
- **vLLM gap narrowed, not closed:** 129–130 tok/s vs vLLM's 221 @VU16 /
  239 @VU32 on the same harness (1.7–1.8× behind, from 2.2–2.3×).
- Attention is serial per request inside the shared launch (R sequential
  flash passes → R partial-wave tails); stage 2's block-diagonal varlen flash
  is the remaining lever in this lane.
- Short prompts pay bucket-covering padding when batched with nothing else
  (a lone 20-token prompt runs a 128-row bucket — same as serialized — but a
  lone 2500-row chunk runs the 4096 bucket where the serialized ladder ran
  [2048, 512]; padding-minimal pack budgeting keeps this bounded to one
  bucket rung).
- The px1-off baseline here (28.7 → 102 tok/s) is faster than the committed
  `plow-b16-bfix` rows (21.9 → 92.9) — different decode cubin (`b16-mm16`)
  and newer branch; the A/B in THIS file is same-binary/same-assets and is
  the controlled comparison.
