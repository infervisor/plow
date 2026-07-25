# Gemma-4-12B served through `plowrt serve` — first real-GPU OpenAI-endpoint numbers (sm_120)

**Campaign S1-serve-gpu.** First end-to-end serving from the plow runtime: the
OpenAI-compatible server computes actual Gemma-4 tokens on the RTX PRO 6000
(sm_120, 188 SMs) through the new `exec::gpu` engine (driver-API port of the
HF-verified `gemma4_sm120_chat.cu` sequence), replacing the FNV-seeded
`reference_logits` stand-in. Decode-only, batch 1; greedy consumes the device
`ARGMAX_FIN` token with no logits download.

## Setup

- **Model:** google/gemma-4-12B-it at `/workspace/models/gemma-4-12B-it` (bf16,
  tied embeddings; 22.2 GiB weights).
- **Packet:** `PLOW_UNISEG=1 plowc gemma4 <dir> 4096 out.pkt 188` (uniseg so the
  standalone harness accepts the prefill buckets; the T=1 decode program is
  byte-identical with or without it — verified by direct comparison).
- **Cubin:** `scripts/build_sm120_cubin.sh` — `-arch=sm_120a -DPLOW_NV_GEMMA=1
  -DPLOW_NV_FA_GF=2` (GQ scheduler default), kernel
  `_Z12interp_sm12011PlowProgram`, dynamic smem 12352 B, grid 188 (= n_cu).
- **Server:** `plowrt serve --assets <dir>` built
  `--release --features cuda,hf-tokenizer`; assets dir = pkt + cubin +
  `tokenizer.json` + `checkpoint` symlink + minimal `weights.json`.
  Engine load (blob parse + 22.2 GiB pinned-staged H2D + tables): **3.1 s** warm.
- **Measurement:** single streaming request, temperature 0, ~3587-token prompt
  (Gemma-4 canonical chat template), 128 generated tokens; SSE frame gaps
  timed client-side; first 16 steps discarded (repo convention); the zero-gap
  finish frame excluded. Harness comparison ran in the same lease on the same
  pkt with the served request's exact prompt ids.

## Correctness gates (all pass)

- `curl /v1/chat/completions` "What is the capital of France?" (greedy) →
  **"The capital of France is Paris."**, `finish_reason: stop` on `<turn|>`.
- **Token parity, short prompt:** served ids `818 5279 529 7001 563 9079 236761
  106` == harness `PLOW_IDS` (PLOW_PREFILL=0), 8/8.
- **Token parity, long prompt (ctx 3587, 128 tokens):** served stream ==
  decode-only harness stream **128/128**; harness device/host argmax AGREE.
  With `PLOW_PREFILL=1` the harness diverges from the decode-only trajectory at
  token 71 — the prefill kernels' different bf16 reduction order flips a
  near-tie; that is a property of the two prompt paths, not a serving defect
  (both runs still 73/128 identical prefix and fluent output).
- `/metrics` sane (token counts, batch size 1, no sheds/deadlocks); streaming
  works; `cargo test -p plowrt` (default features, CPU reference path) green.

## Served TPOT vs the raw harness (same box, same lease, same pkt, same ids)

| path | timed steps | mean ms/tok | median | sd | tok/s |
|---|---:|---:|---:|---:|---:|
| raw harness (decode-only, ctx 3587→3715) | 112 | **18.313** | 18.311 | 0.014 | 54.6 |
| `plowrt serve` OpenAI SSE, single user   | 111 | **18.473** | 18.459 | 0.092 | 54.1 |

**Serving overhead: +0.160 ms/token (+0.87%)** over the raw harness on the same
trajectory — the mux tick, the spawn_blocking hop, the per-token channel send,
SSE framing, and client-side timestamping, all inside 160 µs. The harness's own
split attributes 18.275 ms to launch+sync and 0.037 ms to its host prologue, so
the served number is within 1% of the kernel floor. (The committed raw-harness
reference at input 4096 on the 132k-ctx packet is 18.543 ms/tok; this campaign's
window is ctx 3587–3715, hence the slightly lower 18.31 raw baseline.)

- **TTFT: 65.4 s** for the 3587-token prompt — the decode-only prompt
  consumption (O(n) launches at ~18.2 ms each). This is the known, stated cost
  of the milestone scope; prefill-in-serve is the next task and cuts it ~55×
  (the same-lease prefill-bucket run consumed the same prompt in 2.98 s,
  1202 tok/s).
- Batch is 1 by design; a second concurrent request is rejected at admission
  ("engine at capacity"). batch>1 is a later task.

## Campaign S2-serve-prefill — TTFT via in-serve prefill (this pass)

`plowrt serve` now consumes the prompt through the **prefill bucket chain** on
admission instead of one decode launch per prompt token. On the first mux tick
the sm_120 engine runs the chunked tiled-GEMM + FLASH_PREFILL buckets (the `_pf`
object, `PLOW_NV_PREFILL=1`, 236 regs, 77.5 KiB dynamic smem) over the whole
prompt, leaving the KV cache built and `in.ids` holding the first token — the
exact postcondition of the old decode-only consumption loop, so per-token decode
continues unchanged. Port of the standalone `gemma4_sm120_chat.cu`
`PLOW_PREFILL=1` path (`exec::gpu::GpuEngine::prefill`).

### Setup

- **Packet:** `gemma4-12b-pf.pkt` — `PLOW_UNISEG=1 plowc gemma4`, 128k context,
  single-segment (n_seg=1) prefill buckets `[128, 512, 1024, 2048, 4096, 8192]`
  + the T=1 decode program; GQ appendix present for every program.
- **Cubins (two, both shipped in the assets dir):** `interp_sm120.cubin` (decode,
  150 regs) + `interp_sm120_pf.cubin` (prefill, `-DPLOW_NV_PREFILL=1`, 236 regs,
  the merged **T3** cp.async-pipeline GEMM). `scripts/build_sm120_cubin.sh` now
  emits both. The engine loads the `_pf` cubin as `<assets>/interp_sm120_pf.cubin`
  (`PLOW_NV_CUBIN_PF` overrides); absent, serve falls back to decode-only.
- **Server:** `plowrt serve --features cuda,hf-tokenizer`, batch 1, greedy.
  Engine load (blob + 22.2 GiB weights + decode tables + 6 prefill bucket
  programs) **3.7 s** warm. TTFT measured client-side over the OpenAI SSE endpoint
  (admission → first token frame).

### Correctness (all pass)

- `curl /v1/chat/completions` "capital of France" → **"The capital of France is
  Paris."**, `finish stop`.
- **Parity, short prompt (40 tok):** served greedy stream == standalone
  `gemma4_sm120_chat` `PLOW_PREFILL=1` on the SAME pkt + SAME ids, **48/48**.
- **Parity, long prompt (ctx 3587, 128 tok):** served == harness prefill
  **128/128**. The harness's prefill-vs-decode-only near-tie divergence (bf16
  reduction-order flip) is **at token 71 — the very token the S1 decode campaign
  documented** — and serving reproduces the harness *prefill* trajectory
  exactly (128/128). **Serving introduces no divergence beyond the kernel-level
  one.** The harness was rebuilt at the served cubin's commit so both run the
  identical T3 prefill kernel.

### Served TTFT + TPOT, single user, greedy (one lease, one server)

| target | n_prompt | prefill chunks | **served TTFT** | served TPOT | vLLM TTFT | plow/vLLM |
|---|---:|---|---:|---:|---:|---:|
| 1k  | 1037  | [2048]                    | **0.36 s** | 18.70 ms | 0.117 s | 3.1× |
| 4k  | 4137  | [8192]                    | **1.62 s** | 18.77 ms | 0.323 s | 5.0× |
| 16k | 16437 | [8192,8192,128]           | **5.62 s** | 19.29 ms | 1.502 s | 3.7× |
| 32k | 32837 | [8192,8192,8192,8192,128] | **16.91 s**| 20.00 ms | 2.815 s | 6.0× |
| 64k | 65637 | [8192]×8 + [128]          | **60.77 s**| 21.34 ms | — | — |

- **The 65 s decode-only TTFT is gone.** The committed decode-only served
  baseline was **65.4 s @ 3587 tokens** (O(n) decode launches). In-serve prefill
  serves the comparable 4137-token prompt in **1.62 s — a ~40× TTFT reduction**,
  and 16k / 32k / 64k prompts (which decode-only priming would take ≈ 5 / 11 / 20
  minutes) in 5.6 / 16.9 / 60.8 s.
- **TPOT is unchanged** by the handoff: 18.70 → 21.34 ms/tok over 1k → 64k,
  matching the committed decode sweep's full-KV scaling (S1's 18.47 @ 3587) — the
  prefilled KV cache is bit-consistent with decode's expectation.
- **Bucket-rounding cost.** A prompt just past a bucket boundary pays ~2×
  padded-row compute: 1037→2048 and 4137→8192 (padded ratio ≈ 1.98), while
  16k/32k/64k land near multiples of the 8192 max bucket (ratio ≈ 1.00). This is
  why 1k/4k are the worst points vs vLLM; a 4096-token prompt (exact 4096 bucket)
  primes in ~0.8 s. vLLM chunks to exact length and pays no padding.
- **vs vLLM:** plow's T3 prefill kernel is still 3–6× behind vLLM's TTFT,
  consistent with the T3 raw-prefill ledger (2.6× at exact 4k) plus the ~2×
  padding at the 4k point; the FFMA O(ctx²) flash P·V tail (T4, in progress) is
  the next lever, most impactful at long ctx. This campaign's deliverable is the
  **serving integration** (kill the 65 s decode-only TTFT), not a prefill-kernel
  number.

### Served TTFT vs raw T3 prefill (same box, same lease, same ids)

| n_prompt | raw prefill (harness `PLOW_PREFILL=1`) | served TTFT | serving overhead |
|---|---:|---:|---:|
| 4137  | 1604 ms (2579 tok/s)  | 1620 ms  | **+16 ms (+1.0%)** |
| 32837 | 16825 ms (1952 tok/s) | 16910 ms | **+85 ms (+0.5%)** |

Served TTFT sits **within ~1% of the pure prefill wall time** — the delta is
prompt tokenization + the first decode token + the mux/SSE hop, the same
sub-percent serving overhead the S1 decode-TPOT campaign measured (+0.87%).

- Batch is 1 by design; a second concurrent request is rejected at admission.
- **128k served prefill: not run in-lease** (T3 128k prefill ≈ 290 s, dominated
  by the FFMA flash P·V tail — a T4 target); projects to ≈ 5 min TTFT until T4
  lands. 64k already primes in 60.8 s.

## Campaign S3-served-t4 — the T4 mma P·V prefill through serve (this pass)

Re-measurement of the served path at the T4 merge (rtx `7fe44b8`): the `_pf`
object now carries the **register-resident mma.sync P·V** flash-prefill (236
regs, 83.25 KiB dynamic smem — the engine's new `21312*4` default), the decode
object is the merged batch-ladder build (174 regs). Same pkt
(`gemma4-12b-pf.pkt`, 128k ctx), same method as S2, one lease per phase.

### Correctness (all pass)

- "capital of France" greedy → **"The capital of France is Paris."**, `stop`.
- **Parity, short (48 tok):** served == T4-rebuilt harness `PLOW_PREFILL=1`,
  same pkt + ids, **48/48**. (Harness prefill-vs-decode-only near-tie flip now
  at token 45 with the T4 kernel; serving reproduces the prefill trajectory.)
- **Parity, long (ctx 3587, 128 tok):** served == harness prefill **128/128**.

### Served TTFT + TPOT, single user, greedy (vs S2 and vLLM bf16)

| target | n_prompt | S2 TTFT | **S3 TTFT** | Δ | S3 TPOT | vLLM TTFT | plow/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1k   | 1037   | 0.36 s  | **0.31 s**  | −14% | 18.69 ms | 0.117 s | 2.6× |
| 4k   | 4137   | 1.62 s  | **1.19 s**  | −27% | 18.77 ms | 0.323 s | 3.7× |
| 16k  | 16437  | 5.62 s  | **3.49 s**  | −38% | 19.29 ms | 1.502 s | 2.3× |
| 32k  | 32837  | 16.91 s | **9.66 s**  | −43% | 20.03 ms | 2.815 s | 3.4× |
| 64k  | 65637  | 60.77 s | **34.65 s** | −43% | 21.41 ms | 8.470 s | 4.1× |
| 128k | 131237 | —       | **198.8 s** | —    | 24.25 ms | 16.271 s | 12.2× |

- **TTFT −14%…−43%** vs S2 — the T4 mma P·V replaces the FFMA O(ctx²) flash
  tail, biting hardest where attention dominates (32k/64k). **TPOT is
  unchanged** (the decode program didn't change; matches the committed sweep).
- **128k measured for the first time**: 198.8 s, prefill-bound (~660 tok/s
  amortized at 131k rows — the flash P·V share keeps growing with ctx; the
  next long-ctx prefill lever is attention-side, not GEMM).
- **Raw T4 prefill, same server ids:** 4137 tok in **1175 ms (3521 tok/s)**
  — 5.3× the T3 667 tok/s at this point — and 32837 tok in **9556 ms
  (3436 tok/s)**. Served TTFT is **+1.3% / +1.1%** over raw — the same
  sub-percent-ish serving overhead S1/S2 measured.
- vs vLLM: 2.3–4.1× at 1k–64k (was 3.1–6.0×). The 1k/4k points still pay the
  ~2× bucket-rounding padding; 128k is the flash-prefill scaling gap.

## Campaign S4-served-batch — batch>1 through the serve stack (this pass)

Serving pending #4 closed: `plowrt serve` now drives **B concurrent
sequences** end-to-end. The `exec::gpu` engine drives B sequence slots off a
`PLOW_DECODE_BATCH=B` blob (batch-major KV `[B][kv_head][ring][hd]`; the
decode program is the blob's last, `t == B`): per mux tick, new arrivals
prefill **per-slot** (the prefill programs address the KV cache
slot-relative, so the engine repoints the `kv.*` tensor-table entries at
slot b's ring for the chunk chain, then restores), and every running slot
advances in **one batched cooperative launch** (`ids/pos/kvlen[0..B]`;
`argmax_fin` writes `ids[b]` per sequence). Blobs:
`PLOW_UNISEG=1 PLOW_DECODE_BATCH={1,2,4} plowc gemma4 <12B> 8192 … 188`;
same S3 cubins (the merged decode object already carries the
`gemv_rows<MM>` ladder). Bring-up caught and fixed a compiler defect: every
*prefill* bucket emitted argmax with `n_batch = t` (bucket size!) — a
`t*vocab` OOB logits read on any post-batch-merge blob (`bb302f5`).

### Correctness through the serve stack (all pass)

- **Isolation gate:** 4 (and 2) concurrent OpenAI requests with DIFFERENT
  prompts, greedy — each response **40/40 token-identical** to its own
  single-request run on the same server, engine slots arbitrarily permuted
  (solo runs all use slot 0; concurrent runs landed on shuffled slots). No
  cross-sequence bleed — the kernel-level isolation proof now holds through
  admission → tokenize → per-slot prefill → batched decode → SSE.
  Repeated on the B=2 blob (2/2). "Capital of France" → Paris, `stop`.
- `cargo test -p plowrt` green (63 tests).

### B-scaling, full occupancy, ctx ~4k (4137-tok prompts, 96 tok, greedy)

| blob | users | per-user TPOT | aggregate | scaling | mdq projection |
|---|---:|---:|---:|---:|---:|
| B=1 | 1 | 18.52 ms | **54.6 tok/s**  | 1.00× | — |
| B=2 | 2 | 19.56 ms | **102.3 tok/s** | **1.87×** | 1.89× |
| B=4 | 4 | 21.21 ms | **187.7 tok/s** | **3.44×** | 3.41× |

- Measured aggregate scaling lands **on the mdq projection** (1.87 vs 1.89,
  3.44 vs 3.41): the GEMV weight-read amortization delivers, and the fixed
  per-step cost (attention span, norms, dispatch) caps it exactly as
  modeled. Per-user TPOT degrades only **+5.6% (B=2) / +14.5% (B=4)** — 2
  and 4 users cost each user a barely-perceptible slowdown.
- Under-occupancy on the B=4 blob (its M=4 GEMV cost is compiled in):
  1 user 19.93 ms / 50.7 tok/s, 2 users 20.43 ms / 98.0 tok/s — a B=4
  server serves 1–3 users within 8% of the B-matched blob, so one blob can
  reasonably serve a fluctuating user count.
- Short-ctx (≈30-tok prompts) x4: 197.8 tok/s aggregate, 19.97 ms/user.

### Continuous batching (arrival while decoding, B=4 blob)

A second 4137-token request admitted **mid-decode** of a running stream:
its served TTFT is **1.19 s — identical to the cold single-user TTFT** (its
own prefill; no queueing penalty), while the running stream stalls for at
most **1.21 s = exactly one prefill** (the tick-granularity admission cost
by design: prefill runs inside the mux tick), then resumes at its steady
TPOT. Chunked/interleaved prefill to bound the stall is future work.

## Campaign S5-serve-tuning — serving-path perf audit + chunk-interleaved prefill (this pass)

Systematic audit of the serving path (mux tick loop, engine host ops, channels,
allocations) with a new kernel-only control, then two prefill-policy changes.
Same S4 blobs (`/root/gpu-assets-b4/{b1,b4}`, ctx8k, PLOW_DECODE_BATCH=1/4),
one lease, rc=0 (`/root/gpu-assets-s5/lease1_out.txt`).

### New kernel-only control: `examples/step_bench`

Drives `GpuEngine` directly (no HTTP/mux/spawn_blocking/SSE) — the raw floor
serving should be judged against, at any B. ctx 4137, 128 timed steps:

| config | raw step | served TPOT | serving overhead |
|---|---:|---:|---:|
| B=1 blob, 1 slot  | **18.318 ms** (sd 0.014) | 18.479 ms | **+0.161 ms (+0.88%)** |
| B=4 blob, 4 slots | **20.975 ms** (sd 0.019) | 21.14 ms (mean of 4 users) | **+0.16 ms (+0.8%)** |
| B=4 blob, 1 slot  | 19.681 ms | 19.913 ms | +0.23 ms (+1.2%) |

- **The B=4 per-user +14.5% TPOT cost is entirely kernel**: raw 20.975/18.318
  = 1.145 with zero serving in the loop (batched GEMV per-step growth, exactly
  the mdq byte-model shape). Serving adds the same ~0.16 ms at B=1 and B=4.
- `PLOW_STEP_TIME=1` host-op breakdown per decode step (B=1): kvrow patch
  7.2 us + ids/pos/kvlen upload + single counter/cursor memset 26 us + launch
  4 us + **sync 18.265 ms (the kernel)** + token readback 16 us — host total
  ~53 us inside the step, inter-step gap <1 us in the raw loop. The remaining
  ~0.16 ms served overhead is the mux tick + spawn_blocking hop + SSE framing
  + client-side timestamping, i.e. **the serving stack is within 1% of the
  kernel floor and the floor is the kernel, not the host**.
- Micro-fixes landed (measured-neutral to +): per-step Vec allocs hoisted into
  engine buffers; gq-cursor co-allocated with the counter block (one memset
  per step, was two); dispatcher skips the CPU bucket-ladder scan for
  GPU-engine models; observer no longer reallocated per tick; prompt
  tokenization moved off the dispatcher onto the HTTP handler task;
  incremental windowed detokenize (O(window) per token, was O(total) — ~ms
  per token by 4k-token outputs).

### Prefill chunk policy: minimize padded rows (default; `PLOW_PF_COVER=1` = old)

Old policy ran the smallest single bucket covering the whole remainder — a
4137-token prompt ran ONE padded 8192-row chunk (1.98x rows). New policy runs
the largest bucket that fills completely and rounds up only the tail:
4137 -> [2048, 2048, 128] = 4224 rows.

| metric (4137-tok prompt, B=1 blob) | old (cover) | new (decomposed) |
|---|---:|---:|
| raw prefill (step_bench)  | ~1.18 s | **0.68 s (−42%)** |
| served TTFT               | 1.19 s (S4) | **0.70 s (−41%)** |

Old-policy control re-run in the same lease reproduced 1.19 s TTFT exactly.
1k point (≈1050-tok prompt): served TTFT 0.34 s on the ctx8k blob
(decomposed [1024,128] = 1152 rows vs one 2048 chunk).

### Chunk-INTERLEAVED prefill (arrival mid-decode; the queueing fix)

`GpuEngine::prefill_chunk` + mux: when any slot is mid-decode, at most ONE
capped prefill chunk runs per tick (`PLOW_PF_INTERLEAVE` rows, default 2048;
0 = whole prompt per tick) and the batched decode launch runs between chunks.
The unfed-row garbage KV write stays safe: the engine keeps `pos[b]` at the
prefill frontier, so it always lands in the row the next chunk overwrites.
B=4 blob, A decoding 4137-ctx stream, B (4137-tok) arrives mid-decode:

| config | B's TTFT | A's max stall | A's TPOT over window |
|---|---:|---:|---:|
| S4 baseline (= cover + no interleave, control re-run) | 1.19 s | **1192 ms** | 24.72 ms |
| decomposed only (`PLOW_PF_INTERLEAVE=0`) | **0.73 s** | 631 ms | 22.71 ms |
| decomposed + interleave 2048 (default) | 0.87 s | **426 ms (−65%)** | 23.28 ms |

The default trades +0.14 s arrival TTFT for a 426 ms stall bound (~= one
2048-row chunk + one decode tick); reproducible (426/426 on repeat). Smaller
`PLOW_PF_INTERLEAVE` tightens the bound further at more total prefill time.

### Gates (all pass)

- "capital of France" greedy -> "The capital of France is Paris.", `stop`
  (B=1 and B=4 servers).
- Isolation: 4 concurrent distinct prompts on the B=4 blob, each 40/40
  token-identical to its solo run (CONC_RESULT PASS); slots permuted.
- B-scaling preserved: 50.7 / 97.5 / **187.3 tok/s** at 1/2/4 users
  (S4: 54.6 / 102.3 / 187.7 — the 1/2-user points now include the faster
  TTFT window in the same wall, aggregate at 4 users unchanged).
- `cargo test -p plowrt` green (default + hf-tokenizer features).

### Decisions recorded

- **mdq: dropped** (module lives only on the serving branch, 5e880f6). The
  B<=8 mux is a loss+FIFO system whose waits are prefill-policy-dominated
  (fixed above); deadline-shedding queued requests would break legitimate
  long-decode clients, and abandoned clients are already reaped by implicit
  cancellation (now also mid-prefill). EWMA lambda stays for the cold-start
  formation window; the per-tick admit() Shed branch remains inert by design.
- Launch/readback overlap (double-buffered token page) is design-only: in.ids
  is both argmax output and next-step input, so overlapping needs a
  kernel-side parity buffer — bounded by the measured ~53 us host share,
  <0.3% of the step; not worth kernel churn.

## Campaign S6-refresh — assets on HEAD cubins + the served 128k point (this pass)

Asset refresh at `main` @ `a058492`: the serve cubins were T4-era; rebuilt via
`scripts/build_sm120_cubin.sh` at HEAD — decode **212 regs** (the 638ce37
batched bf16/fp8 GEMV arms), prefill `_pf` **240 regs** (T5+T8 pipeline,
smem_pf 81664, `PLOW_NV_EMBED_SMEM` metadata), 0 spills both. New 128k serving
dir **`/root/gpu-assets-s6/b1`** (pkt re-emitted at HEAD: `PLOW_UNISEG=1
gemma4 <12B> 132096`, buckets [128..8192]+decode); `/root/gpu-assets-b4`'s
b1/b2/b4/b8 cubin symlinks repointed to the HEAD pair (ctx8k batch blobs
unchanged).

### Gates (all pass)

- "capital of France" → **"The capital of France is Paris."**, `stop`.
- Parity: served greedy == HEAD-rebuilt harness `PLOW_PREFILL=1`, same pkt +
  ids, **48/48** (prefill-vs-decode-only near-tie flip @45, the documented
  kernel class).

### Served TTFT + TPOT, single user, greedy (vs vLLM bf16)

| target | n_prompt | S3 TTFT (T4-era) | **S6 TTFT** | S6 TPOT | raw prefill (same lease) | vLLM TTFT | plow/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4k   | 4137   | 1.19 s  | **0.59 s**  | 18.92 ms | 949 ms (cover) | 0.323 s | 1.8× |
| 32k  | 32837  | 9.66 s  | **5.21 s**  | 20.12 ms | 5115 ms | 2.815 s | 1.9× |
| 128k | 131237 | 198.8 s | **37.79 s** | 24.32 ms | 37573 ms | 16.27 s | 2.3× |

- **The T4-era 198.8 s served 128k TTFT row is retired: 37.79 s (−81%)** —
  within **+0.6%** of the raw T5+T8 prefill wall (37.57 s), landing on the
  capacity report's ~37.6 s projection. 32k overhead +1.9%; the 4k served
  point *beats* the raw cover-policy number because serve decomposes
  4137 → [4096,128] = 4224 rows vs the harness's single 8192 bucket.
- TPOT unchanged vs the committed sweeps (18.92/20.12/24.32 across
  4k/32k/128k; engine-direct `step_bench` same lease: 18.79/19.99/24.19).
- **TTFT convention note:** the server now emits a role-ack SSE frame at
  admission; TTFT is measured to the first *content* frame (the old client
  counted the ack and read ~20 ms).
- **Serving regression found (worked around, not fixed — serving layer owned
  elsewhere):** `f6e635c` activated the admission shed with a per-TICK
  `service_ms` EWMA. One long prefill tick (> `--slo-ms`, default 250) poisons
  the estimate and the next tick sheds EVERY live slot as "arrival-rate
  admission shed" — a 32k prompt gets 1 token then a silent `finish: stop`
  (the mux `Err` path also swallows the message). All S6 measurements run
  `--slo-ms 100000000`. Proper fix: exclude prefill ticks from `service_ms`
  (or compare per-request predicted wait), and log the shed/Err reason.

### 31B + 26B served (same pass)

31B and 26B are now served models with their own asset dirs and committed
rows — see `gemma4-31b-plowrt-served.{json,md}` and
`gemma4-26b-plowrt-served.{json,md}`. Headline: 31B TPOT@32k **48.78 ms
beats vLLM 49.14**; 26B TPOT@32k **9.22 ms beats vLLM 9.57**; both pass
Paris + 48/48 capture-replay parity.
