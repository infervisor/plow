# PX-5 — cudaGraph prefill chunk loop — **NO-GO** (go/no-go memo)

RTX PRO 6000 Blackwell (sm_120a), gemma-4-12B bf16, TP1. Branch `px3-px5-cheap-checks`.

**Verdict: NO-GO — do not build.** The maximum possible short-ctx TTFT saving is **< 0.03%**,
~70–400× below the ~2% go threshold. The go/no-go arithmetic below killed it; graph
capture/replay was **not** implemented (a valid, valuable cheap-check outcome).

## The premise is structurally false

The plan hypothesised "short ctx / many small chunks." plow's prefill chunk loop
(`gemma4_sm120_chat.cu`) picks the **smallest bucket ≥ remaining tokens** from
`[128, 512, 1024, 2048, 4096, 8192]` (`MAX_CHUNK=8192`). So **any prompt ≤ 8192 tokens is a
SINGLE prefill chunk = a SINGLE cooperative launch.** "Many small chunks" never happens at short
ctx. Multi-launch only begins **above** 8192 tokens (long ctx), where TTFT is seconds and the
per-launch tax is even more negligible. **There is no context regime where a chunk-loop graph is
material.** cudaGraph's value scales with launch count; at 1 launch there is nothing to amortise.

## Measured inputs

| input | value | source |
|-------|-------|--------|
| chunks at 512 / 2048-tok prompt | **1 / 1** | harness run (single `PREFILL:` line) |
| per-token host prologue | **44 µs** | `gemma4_sm120_chat` decode breakdown, this box (matches plan's 53 µs) |
| per-launch cooperative-launch enqueue | **3.08 µs** | T9c (`299 µs / 97 launches`) |
| prefill chunk host prologue (upper bound) | **~80 µs** | 44 µs + one full instruction-stream `cudaMemcpy` |
| prefill_ms @ ctx 512 | **415.87 ms** | harness `PLOW_PREFILL_RESULT` |
| prefill_ms @ ctx 2048 | **1499.33 ms** | harness `PLOW_PREFILL_RESULT` |

## The ceiling

A graph replay removes the **host-side issue** of the per-chunk counter-zero (`cudaMemset`),
kv-row patch (`cudaMemcpy`), and the cooperative launch — **not** the device compute. At 1 chunk
that is at most **one** prefill host prologue (≤ ~80 µs).

| ctx | prefill_ms | max removable | **ceiling** |
|-----|-----------|---------------|-------------|
| 512  | 415.87 | ~0.08 ms | **0.019 %** |
| 2048 | 1499.33 | ~0.08 ms | **0.005 %** |

Go threshold: **2 %**. The ceiling is ~70–400× under it.

Independent cross-check: the decode-step breakdown on this box is host `0.044 ms` vs launch+sync
`18.07 ms` — host issue is **0.24 %** of even a single decode step, and a prefill chunk's kernel is
far larger (hundreds of ms), so its host fraction is smaller still.

## Why not built

Graph capture/replay of the chunk sequence needs per-chunk graph-node param updates (the
counter-zero memset and the kv-row-patch memcpy change every chunk) plus re-capture on bucket-shape
change — real complexity for a ≤ 0.02 %-of-TTFT saving. The plan anticipated this ("the per-token
host prologue is already 53 µs, so the ceiling here is small"), and T9c already measured the host
enqueue tax as "negligible vs the ~514 ms GEMM-bound compute." The prefill wall is GEMM/flash
**compute**, which PX-1 (batched prefill) and PX-2 (fp8 mainloop) attack — PX-5 does not.

## Artifacts

- `/tmp/px5_512.out`, `/tmp/px5_2048.out` — raw harness runs (transient).
- Inputs: T9c `perf-data/gemma4-12b-t9c-segments-sm120.json` (3.08 µs/launch), harness decode
  breakdown (44 µs host prologue).
