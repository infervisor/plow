# Gemma-4-31B TENSOR-PARALLEL PREFILL — plow, TP=1/2/4/8 (branch `tp-prefill`)

Closes plow's one structural weakness: prefill used to run SINGLE-GPU regardless of TP degree
(`tp_decode` was decode-only; `gemma4.rs` emitted decode-only sharded blobs for tp>1), so plow lost
long-context prefill/TTFT to vLLM by 4–22× (`gemma4-31b-longctx-sweep.md`). This work shards prefill
across N GPUs with the SAME Megatron machinery decode already uses.

Measured 2026-07-17, 8× MI350X / gfx950, bf16, greedy, chunk 8192. Model
`/home/lava/models/gemma-4-31B-it-text`. `sg render`, clean env.

## 1. Bit-exactness GATE — PASSED at TP=1/2/4/8

The sharded GEMMs + the [T,hidden] all-reduce (= the full GEMM) + head-split flash_prefill are
exact. First generated token is TOKEN-IDENTICAL across every TP degree, at every context, and
device==host argmax holds on every cell (all ranks agree):

| ctx  | first token (TP1 = TP2 = TP4 = TP8) |
|------|-------------------------------------|
| 8k   | 29486 |
| 32k  | 20606 |
| 64k  | 28766 |
| 128k | 19841 |

## 2. Prefill TTFT (ms) and throughput (tok/s)

| ctx  | TP1 ms | TP2 ms | TP4 ms | TP8 ms | TP1 tok/s | TP2 tok/s | TP4 tok/s | TP8 tok/s |
|------|-------:|-------:|-------:|-------:|----------:|----------:|----------:|----------:|
| 8k   | 1174.5 |  810.7 |  566.7 |  547.7 |  6 975 | 10 104 | 14 455 | 14 958 |
| 32k  | 7586.2 | 4679.4 | 2998.3 | 2570.7 |  4 319 |  7 003 | 10 929 | 12 746 |
| 64k  |22773.4 |13174.8 | 7881.0 | 6099.2 |  2 878 |  4 974 |  8 316 | 10 745 |
| 128k |75850.6 |42019.8 |23321.4 |16010.7 |  1 728 |  3 119 |  5 620 |  8 186 |

- **TP1 reproduces the shipped single-GPU baseline** (128k 75850 ms vs the longctx-sweep's 75673 ms).
- **Scaling grows with context** (the shardable weight-stream + O(T²) attention rise with T): TP8
  speedup over TP1 is 2.14× @8k → 2.95× @32k → 3.73× @64k → **4.74× @128k**. TP4: up to 3.25× @128k.

## 3. Does TP prefill close the gap to vLLM?

vLLM TP4 prefill (from `gemma4-31b-longctx-sweep.md`, reliable points): 96k 25 001 tok/s,
128k 20 056 tok/s.

| ctx  | plow before (TP1) | plow now (TP8) | vLLM TP4 | gap: before → now |
|------|------------------:|---------------:|---------:|------------------:|
| 128k | 1 728 tok/s (11.6× slower) | 8 186 tok/s (2.45× slower) | 20 056 tok/s | **11.6× → 2.4×** |

**VERDICT.** TP prefill closes most of the structural gap: from **11.6× slower** (single-GPU) to
**~2.4×** at 128k (TP8). plow does not yet *win* prefill outright, but the single-GPU-prefill
weakness is eliminated and prefill now scales across N GPUs. Combined with plow's decode lead
(branch `tp`, +11 % TPOT @128k, bit-exact), plow is now competitive in BOTH phases at long context.

## 4. The [T,hidden] collective (the new regime)

Prefill activations are [T, hidden] (T tokens), so the two all-reduces/layer (after o_proj and down)
are **T× bigger** than decode's [1, hidden] — bandwidth-bound, not latency-bound. The current
implementation reuses the decode one-shot XReduce (each rank sums all N peers' full partial). It is
correct and bit-exact, but O(N) fabric per rank — which is why TP8 scaling falls off from the ideal
8× to 4.74×. The bandwidth-optimal choice here is a **reduce-scatter + all-gather (two-shot)**:
~2·(N−1)/N·T·H·2 B/rank vs the one-shot's N·T·H·2 B/rank (~N/2× less fabric). That is the top
remaining lever, together with cutting the per-segment host barriers and the replicated
lm_head/embed/norms.

## 5. Reproduce

Compile (fast — no weight load): `plowc gemma4 --tp {1,2,4,8} <model> 131072 g131k_tpN.pkt`.
Run: `./tp_decode g131k_tpN.pkt <model> --tp N --prefill synth --pf-sweep 8k,32k,64k,128k --chunk 8192`
under `sg render` + clean env (`/usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME LD_LIBRARY_PATH=/opt/rocm/lib`),
with `interp_prefill.elf` + `interp_flash.elf` (+ `interp_decode_gq.elf`) in cwd. Bit-exactness: the
`bitexact` column reads `OK(dev==host,ranks agree)` and the `tok0` column is identical across TP.
