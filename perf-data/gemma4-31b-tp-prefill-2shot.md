# Gemma-4-31B TWO-SHOT TP PREFILL all-reduce — plow, TP=1/2/4/8 (branch `tp-prefill-2shot`)

Replaces the one-shot [T,hidden] PREFILL all-reduce (`gemma4-31b-tp-prefill.md`) with a
bandwidth-optimal **two-shot (reduce-scatter + all-gather)**. The one-shot has every rank read
all N peers' FULL partial — O(N) fabric/rank — which is optimal for decode's tiny [1,hidden]
message but caps TP8 PREFILL scaling because the [T,hidden] partial is bandwidth-bound. Two-shot
moves ~2(N−1)/N·msg/rank instead of (N−1)·msg (~N/2× less fabric at large N).

DECODE keeps the one-shot (its [1,hidden] message is latency-bound — 1 sync beats 2).

Measured 2026-07-17, 8× MI350X / gfx950, bf16, greedy, chunk 8192. Model
`/home/lava/models/gemma-4-31B-it-text`. `sg render`, clean env. TP1/2/4/8 on canonical GPUs
(TP1=GPU0, TP2=0-1, TP4=0-3, TP8=0-7).

## 1. Bit-exactness GATE — PASSED at TP=1/2/4/8

Two-shot is mathematically identical to one-shot (same f32 accumulate, r=0..N−1 order, rounded to
bf16) so the first generated token is TOKEN-IDENTICAL across every TP degree and context, and
device==host argmax holds on every cell (all ranks agree):

| ctx  | first token (TP1 = TP2 = TP4 = TP8) |
|------|-------------------------------------|
| 8k   | 29486 |
| 32k  | 20606 |
| 64k  | 28766 |
| 128k | 19841 |

Identical to the one-shot's tokens — the algorithm change is invisible to the numerics.

## 2. Prefill TTFT (ms) — TWO-SHOT

| ctx  | TP1 ms | TP2 ms | TP4 ms | TP8 ms | TP1 tok/s | TP2 tok/s | TP4 tok/s | TP8 tok/s |
|------|-------:|-------:|-------:|-------:|----------:|----------:|----------:|----------:|
| 8k   | 1166.1 |  827.4 |  533.0 |  415.5 |  7 025 |  9 901 | 15 369 | 19 716 |
| 32k  | 7540.4 | 4766.1 | 2845.2 | 2030.5 |  4 346 |  6 875 | 11 517 | 16 138 |
| 64k  |22654.9 |13409.9 | 7605.2 | 5017.3 |  2 893 |  4 887 |  8 617 | 13 062 |
| 128k |76971.1 |42161.6 |22959.5 |13831.1 |  1 703 |  3 109 |  5 709 |  9 477 |

TP1 reproduces the single-GPU baseline (128k 76 971 ms vs one-shot's 75 850 ms, ~1.5 % noise) —
TP1 has no collective, so it is byte-identical code.

## 3. TWO-SHOT vs ONE-SHOT — the TP8 recovery

TP8 TTFT (ms), two-shot vs the one-shot baseline (`gemma4-31b-tp-prefill.md`):

| ctx  | one-shot TP8 ms | two-shot TP8 ms | faster | one-shot tok/s | two-shot tok/s |
|------|----------------:|----------------:|-------:|---------------:|---------------:|
| 8k   | 547.7  | **415.5**  | **24 %** | 14 958 | **19 716** |
| 32k  | 2570.7 | **2030.5** | **21 %** | 12 746 | **16 138** |
| 64k  | 6099.2 | **5017.3** | **18 %** | 10 745 | **13 062** |
| 128k |16010.7 | **13831.1**| **14 %** |  8 186 | **9 477**  |

Consistent win at every context (two independent TP8 runs agreed within 1 %). The win GROWS as
context SHRINKS — expected: the all-reduce is per-CHUNK [8192,hidden] (constant size), so its share
of prefill time falls as the O(T²) full-attention compute rises with context. At 8k (one chunk,
little attention) the collective is a big fraction → 24 % end-to-end; at 128k (16 chunks, attention
dominates) it is a small fraction → 14 %.

TP8 scaling over TP1 (same build): **2.14× → 2.81× @8k, 2.95× → 3.71× @32k, 3.73× → 4.51× @64k,
4.74× → 5.57× @128k.** Two-shot recovers ~0.7–0.8× of the lost scaling at every context.

TP4: two-shot 533.0 / 2845.2 / 7605.2 / 22959.5 ms vs one-shot 566.7 / 2998.3 / 7881.0 / 23321.4 —
1.6–6 % faster (2(N−1)/N = 1.5·msg vs 3·msg = 2× less fabric, smaller end-to-end share than TP8).

TP2: two-shot 827.4 / 4766.1 / 13409.9 / 42161.6 ms vs one-shot 810.7 / 4679.4 / 13174.8 / 42019.8 —
neutral-to-marginally-slower (~1 %). EXPECTED: at N=2, 2(N−1)/N = 1.0·msg = the one-shot's (N−1) =
1.0·msg — no bandwidth advantage, and two-shot pays one extra rendezvous. The two-shot only wins for
N≥4, exactly where the theory says.

## 4. vs vLLM prefill (long context)

vLLM prefill throughput (`gemma4-31b-longctx-sweep.md`): TP4 128k 20 056 tok/s; TP8 128k 33 498 tok/s.

| ctx  | plow two-shot TP8 tok/s | vLLM TP4 tok/s | plow/vLLM-TP4 | vLLM TP8 tok/s | plow/vLLM-TP8 |
|------|------------------------:|---------------:|--------------:|---------------:|--------------:|
| 128k | 9 477 | 20 056 | 2.12× slower | 33 498 | 3.53× slower |

At 128k, two-shot closes the gap to vLLM TP4 from the one-shot's **2.45× → 2.12×**. It does NOT make
plow win prefill at long context: the remaining 128k gap is NOT the collective (which is now cheap)
but the structural cost — plow re-streams weights per chunk and runs O(T²) single-shard flash on the
10 full-attention layers, while vLLM's sharded prefill amortises weights across shards. The all-reduce
fabric was the dominant TP8-scaling loss only at SHORT context, and that is exactly where two-shot
recovers the most.

## 5. Implementation

- New fused, self-contained device op `PLOW_DOP_XREDUCE2` / `DevOp::XReduceTwoShot` (opcode 29),
  body `d_xreduce_twoshot_mega` in `runtime/amd/op_collective.h`, dispatched in `interp.hip`
  alongside the one-shot `d_xreduce_mega`. Partitions the flat [n=t·hidden] result into N contiguous
  slices; this rank reduces its OWNED slice from all peers (writing it IN-PLACE into its own
  peer-visible partial slot — safe because slice s is read only by rank s), then all-gathers every
  peer's reduced slice into the local full vector. Two internal xctr rendezvous (reduce-scatter,
  all-gather); each collective consumes 2 gate ids instead of 1 (`XCTR_BYTES` bumped 256→512).
- `gemma4.rs` emits `XReduceTwoShot` for PREFILL (`decode==false`) and keeps one-shot `XReduce` for
  DECODE, via the shared `emit_xreduce` helper. No peer-layout change (same partial_A/partial_B slots).
- Register budget unchanged: prefill interpreter still VGPR 256 / occ-2 / spill 0.

## 6. Reproduce

Compile: `plowc gemma4 --tp {1,2,4,8} <model> 131072 g131k_tpN.pkt`.
Run: `./tp_decode g131k_tpN.pkt <model> --tp N --prefill synth --pf-sweep 8k,32k,64k,128k --chunk 8192`
under `sg render` + clean env (`/usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib`),
with `interp_prefill.elf` + `interp_flash.elf` (+ `interp_decode_gq.elf`) in cwd. Bit-exactness: the
`bitexact` column reads `OK(dev==host,ranks agree)` and `tok0` is identical across TP.
