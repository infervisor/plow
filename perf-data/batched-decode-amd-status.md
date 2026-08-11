# Batched decode on AMD — the gate now exists, and it PASSES

Status as of 2026-07-28. Supersedes the 2026-07-27 revision of this file, which said the
correctness gate did not exist and that no B>1 number could be quoted. Both are now false.
The 2026-07-27 text is kept in §4 because its reasoning about *why* the old check proved
nothing is still the right reasoning.

Tree: `/home/lava/plow/build-amd/` (canonical). Model: Gemma-4-31B-it bf16, gfx950, 1 GPU,
`max_ctx` 16384. Blobs `g31b-db{1,2,4,8}`, code objects `hsaco-b{1,2,4,8}` — one pair per
`PLOW_DECODE_BATCH`, both halves of the knob matched (`plowc` sizes the KV cache and the
decode program's `t`; `scripts/build_gfx950.sh` sets `PLOW_GEMV_MM`).

## 1. THE GATE. Every slot reproduces its own prompt's batch-1 stream, token for token.

`plowrt amd-bench --batched --prompt 'p1;p2;p3;p4'` now prefills **one prompt per sequence
slot** (`;`-separated, cycled) via `AmdEngine::prefill_slot`, so every slot's KV is genuinely
populated by this run. Prompts of different LENGTHS make the positions **ragged**, which is
the case a lockstep batch cannot reach. 4 decode steps, greedy:

| prompt (len) | B=1 alone | B=4 slot | B=8 slots |
|---|---|---|---|
| `2,106,1645` (3) | `537, 236789, 64066, 1270, 537` | slot 0 ✅ | slots 0, 4 ✅ |
| `2,106,1645,236764,3689` (5) | `563, 506, 4954, 1534, 496` | slot 1 ✅ | slots 1, 5 ✅ |
| `2,3689,506,7534,529,6427,236761` (7) | `5067, 1852, 1852, 1852, 1852` | slot 2 ✅ | slots 2, 6 ✅ |
| `2,106,1645,236764` (4) | `840, 506, 496, 496, 496` | slot 3 ✅ | slots 3, 7 ✅ |

Both halves of the ask hold: **B copies of one prompt give one stream** (all four slots
`537, 236789, 64066, 1270, 537`), and **B different prompts each give what they give alone**.
This is exactly the gate §4 said "would actually settle it", and it is a strictly stronger
check than the retracted one because it also compares against the B=1 blob.

**A false failure worth knowing about.** The first run of this gate reported
`same-prompt slots agree: NO`, and it was the HARNESS, not the engine. Prefill is
single-sequence, so it writes `in.ids[0]` only; relying on the device's per-sequence argmax
(commit c50472f) after a round of per-slot prefills leaves rows 1.. holding whatever the LAST
prefill left. Every slot but one then decoded from a stale id, which looks precisely like
"per-sequence KV rows are wrong". `amd-bench` and `AmdServe` now both seed all B rows
explicitly. If you see this signature again, check the seed before the KV.

## 2. Per-slot prefill: a pointer rebase, not a second program

The prefill program is single-sequence and stays that way. Its `HeadNormRope` runs with
`n_batch_kv == 0`, so it writes at `hh * out_stride + row` — the *first* sequence's block
relative to whatever base the tensor-pointer table hands it. The KV cache is allocated
`[batch][kv_head][ring][hd]`, so sequence `s`'s block is exactly `s * bytes/batch` in.

`AmdEngine::kv_rebase(s)` therefore edits the `kv.*` entries of the device tensor table,
runs the ordinary prefill, and restores. 120 buffers on this model, 16 MiB per slot,
one table upload per prefill, off the per-token path. `decode_step_batched` refuses to run
with the base left rebased, because that would funnel all B sequences into one slot's cache.

## 3. Ragged positions were ALREADY supported — the refusal was stale

`decode_step_batched` used to reject ragged `pos` on the grounds that the KV write row is one
host-patched immediate (`i[3]`). That is only true of a `batch == 1` program. Since commit
5d600e8, `devgen` arms `i[6] = n_batch_kv` on every decode `HeadNormRope` when `t > 1`, and
`op_norm.h` then takes **both** the write row and the RoPE angle from `pos[t]`:

    obase = (t*nhead + hh) * out_stride + (pos[t] & kv_mask)      // KV row
    p     = pos[t] * H2                                            // RoPE angle

while `d_flash_decode` reads `kv_len[b]` and bases K/V/Q at `b * n_kv_head`. Every
position-dependent term was already per-sequence; nothing about a common `pos` was
load-bearing. `i[3]` is dead on that arm, so `patch_kvrow` is now skipped at `b > 1` rather
than fed a value the kernel ignores.

## 4. Why the OLD `identically-seeded sequences agree` line proved nothing (2026-07-27, still true)

The KV cache is *deliberately never zeroed* (`crates/plowrt/src/exec/amd.rs`):

> *"Attention reads only [0, kvlen), every row of which is written before it is read, so the KV
> cache needs no zeroing — 11.5 GiB of memset skipped on this model."*

Sound **for real serving**, where prefill writes the KV before decode reads it. It does **not**
hold for `amd-bench --ctx N`, which decodes *from* position N **without ever prefilling it**.
Attention then reads N rows of whatever was left in VRAM — and at B>1 each sequence reads a
*different* uninitialised region. So identically-seeded sequences could disagree for reasons
that have nothing to do with batching, and could agree merely because a previous run left the
same pattern in all B regions. Both were observed. **Neither a 15/15 green nor a single
failure from that check was ever evidence about batching.**

That line still prints when `--prompt` is omitted, now labelled as not a gate. Use `--prompt`.

## 5. The device-level ceiling — batching DOES amortise, sub-linearly

`plowrt amd-bench --batched`, 64 dispatches, median of the run, one leased GPU. Blob and code
object matched at each B. **This is an `amd-bench` number: per §0-BENCH it is a bring-up
instrument and must never sit beside a vLLM number.** It is here to answer one question —
does aggregate throughput scale with B at the device level — and it does.

| B | ctx 1024 tpot ms | aggregate tok/s | vs B=1 | ctx 4096 tpot ms | aggregate tok/s |
|---|--:|--:|--:|--:|--:|
| 1 | 17.26 | 57.9 | 1.00x | 18.41 | 54.3 |
| 2 | 18.78 | 106.5 | **1.84x** | 20.01 | 99.9 |
| 4 | 28.23 | 141.7 | **2.45x** | 28.14 | 142.2 |
| 8 | 39.54 | **202.3** | **3.49x** | 40.53 | 197.4 |
| 16 | 112.37 | 142.4 | 2.46x — **a LOSS** | — | — |

3.5x of aggregate throughput for 2.3x of per-token latency at B=8. The discontinuity is
between B=2 and B=4, not B=4 and B=8 — the opposite of the sm_120 shape in
`perf-data/px10-batched-decode.md` §1, whose B=4→B=8 cliff was a build flag (`GV_MM_MAX=16`)
that has no counterpart here.

**B=8 is the top of the curve. B=16 is slower in aggregate than B=8** (142 vs 202 tok/s) and
2.8x worse per token, so the useful range on this model is B ≤ 8. Two caps sit just above it:
`PLOW_GEMV_MAXM = 16` (§8), and the LDS arena (below).

**The latency cost is real and it is the reason B is a deployment choice, not a default.**
A B=8 blob computes 8 rows even when one slot is fed: it is 2.02x slower at concurrency 1
than a B=1 blob through the same endpoint (36.31 vs 17.98 ms TPOT, §6g-LEGAL). There is no
runtime narrowing — the decode program's `t` is compiled, not passed.

## 5a. A BUG THE GATE CAUGHT: `GemvQkv` fused past the LDS arena at B=16

Before the fix below, B=16 produced **fluent-but-wrong** streams on sequences 13, 14 and 15
and token-identical ones on 0..12 — deterministic, reproducible, and invisible to a
seq0-only check. The arithmetic is exact:

| | halves |
|---|--:|
| `GM_LDS_HALVES` = `2*(256+256)*(64+8)` | 73,728 |
| `M*K` at B=16, Gemma-4-31B `hidden` 5376 | 86,016 |
| rows that fit (`73728 / 5376`) | **13** |

Row `m` is staged at `lds[m*K ..]`, so rows 13, 14, 15 run past the end of the arena.

`op_gemm.h` states the precondition on both fused GEMVs — *"x is ALWAYS staged in LDS here:
plowc emits this op only when M*K fits GM_LDS_HALVES"* — and `gemv_qkv_rows` reads x only
through `ld_lds8`; unlike `gemv_rows_fp8<MM, XLDS>` it has **no global-read arm** to fall back
to. `devgen` enforced that precondition for `DevOp::GemvGlu` (`glu_fused`) and **not** for
`DevOp::GemvQkv` (`fuse_qkv` tested `gemv_family && !keqv && !fp8` only). §4's bug shape once
more: a documented precondition with nothing checking it.

**Fixed** by giving `fuse_qkv` the same `(t * hidden) <= GM_LDS_HALVES` test `glu_fused`
already had. Below the threshold the guard is inert — B=1 and B=8 re-emit at 676 packets,
unchanged — and at B=16 q/k/v unfuse (796 → 896 packets) and every slot becomes
token-identical to its batch-1 run.

**This is what the gate is for.** The check that found it had to compare each slot against the
first slot carrying *its* prompt; comparing every slot against slot 0 only tests one prompt
class and reported a green while three slots were wrong.

## 6. THE SERVED CURVE — §0-BENCH-legal, and the shape INVERTS

`vllm bench serve --backend openai-chat` against a plowrt endpoint, `scripts/bench_plowrt_serve.sh`,
random 1024-in / 128-out, 64 prompts per point, one leased GPU, Gemma-4-31B-it bf16 TP1.
Same client binary at every point. **64/64 requests succeeded at every point** — no 429s, so
none of these rows is the "shed reported as success" artifact §6g-LEGAL warns about.

| conc | plow B=1 out tok/s | plow B=8 out tok/s | **B=8 / B=1** | B=1 TPOT ms | B=8 TPOT ms |
|---|--:|--:|--:|--:|--:|
| 1 | **50.2** | 26.4 | 0.53x | 18.23 | 36.31 |
| 4 | 46.4 | **87.5** | **1.89x** | 19.89 | 41.50 |
| 16 | 45.8 | **147.4** | **3.22x** | 20.21 | 46.59 |
| 64 | 47.8 | **146.8** | **3.07x** | 19.26 | 46.77 |

**The B=1 column is the bug this work fixes, reproduced exactly.** Throughput *falls* from
conc 1 to conc 4 and then sits flat at ~46 tok/s however many clients arrive — concurrency is
correct but serialized, which is what §6g-LEGAL recorded as "at conc 4 throughput FALLS to
42.5 tok/s". Its conc-1 TPOT of 18.23 ms also reproduces §6g-LEGAL's 17.98 ms, so the two
measurements are on the same footing.

**The B=8 column is the point.** Throughput now *rises* with concurrency and saturates at
147 tok/s — 3.2x the serialized path — and saturation is at 8 slots, exactly where it should
be: conc 64 adds queueing, not throughput.

Two costs, both real and both structural:

* **Concurrency 1 is 1.9x slower on the wide blob** (36.31 vs 18.23 ms TPOT). Every dispatch
  computes all 8 rows; there is no runtime narrowing. **`PLOW_DECODE_BATCH` is a deployment
  choice that must match the expected load**, and serving a wide blob to a single user is a
  straight loss.
* **TTFT degrades under load** — 233 ms at conc 1, 580 ms at conc 4, 7.1 s at conc 16, 25 s at
  conc 64 — because prefill is neither chunked nor interleaved: one prefill holds the whole
  device and stalls every decoding slot. Past 8 in-flight requests the rest queue in the mux
  channel, and that queue is the whole of the conc-64 TTFT. This is the next lever, and it is
  the same lever the CUDA path already pulled (chunked prefill, `serve/mux.rs`).

Device-level B=8 was 202 tok/s (§5) against 147 served, so the serve layer costs ~27% here.

### 6a. Against vLLM. Batching is necessary and NOT sufficient — plow still loses, badly.

Same client binary, same chat backend, same 1024/128 shape, same box, one GPU each.
vLLM `rocm/vllm:rocm7.14.0 ... vllm_0.23.0`, bf16, TP1 — `scripts/bench_vllm_chat.sh`.

| conc | plow best out tok/s | vLLM out tok/s | **vLLM / plow** | plow TPOT | vLLM TPOT |
|---|--:|--:|--:|--:|--:|
| 1 | 50.2 *(B=1)* | 70.1 | **1.40x** | 18.23 | 13.62 |
| 4 | 87.5 *(B=8)* | 251.9 | **2.88x** | 41.50 | 15.52 |
| 16 | 147.4 *(B=8)* | 679.9 | **4.61x** | 46.59 | 19.95 |
| 64 | 146.8 *(B=8)* | 1999.2 | **13.6x** | 46.77 | 29.77 |

**Read this honestly.** Batching turned a curve that FELL with concurrency into one that
rises, and it is a prerequisite for ever competing at load — but it does not close the gap,
and the gap *widens* with concurrency: 1.4x at conc 1, 13.6x at conc 64. vLLM's TPOT rises
only 13.6 → 29.8 ms from conc 1 to 64 while serving 64 sequences; plow's ceiling is 8.

The three structural reasons, in order of size:

1. **B is compiled, not dynamic, and it tops out at 8** (§5). vLLM schedules however many
   sequences fit in its paged cache. This is the whole conc-16→64 difference: plow's
   throughput is flat past 8 clients because slots 9.. are queued, not batched.
2. **No paged KV.** plow's cache is one flat `[B][kv_head][ring][hd]` allocation sized at
   emit — 22.5 GiB at B=8 / 16k ctx, 45 GiB at B=16 — so B is bounded by VRAM at the *worst
   case* context, not by what the live requests actually use. Paging is what lets vLLM run
   B=64. This is the single highest-value follow-up.
3. **No chunked prefill.** One prefill holds the whole device, so TTFT under load is set by
   the prompts ahead of you in the queue (25 s at conc 64).

`AmdServe` gained slots; it did not gain a scheduler. (1) and (2) are one piece of work —
a runtime-variable batch over a paged cache — and until it exists the conc-64 column will not
move.

## 7. Reproduce

    # code objects, one dir per batch — OUTSIDE nix (knob-contract §0a)
    for B in 1 2 4 8; do
      /usr/bin/env -i PATH=/opt/rocm/bin:/usr/bin:/bin HOME=$HOME PLOW_DECODE_BATCH=$B \
        bash scripts/build_gfx950.sh /home/lava/plow/build-amd/hsaco-b$B
    done
    # blobs, one per batch — INSIDE nix
    for B in 1 2 4 8; do
      PLOW_DECODE_BATCH=$B ./target/release/plowc --hf-dir <gemma-4-31B-it> \
        --emit devblob --arch gfx950 --gpu mi355x --n-cu 256 --max-ctx 16384 \
        --out /home/lava/plow/build-amd/g31b-db$B
    done
    # the gate
    perf-data/tools/gpulease -n 1 gate sg render -c 'nix develop -c \
      ./target/release/plowrt amd-bench --blob build-amd/g31b-db4/model.pkt \
        --hsaco build-amd/hsaco-b4 --checkpoint <ckpt> --steps 4 --ctx 1024 --batched \
        --prompt "2,106,1645;2,106,1645,236764,3689;2,3689,506,7534,529,6427,236761;2,106,1645,236764"'

## 8. Still open

- **`PLOW_DECODE_BATCH > 16` is unservable and `plowc` still emits it.** The decode GEMV is a
  compile-time row bucket: `PLOW_GEMV_MAXM 16`, and `gemv_rows<MM>` carries `float acc[MM]`
  and loops `m < MM` with no outer loop over `M > MM`. `scripts/build_gfx950.sh` clamps
  `PLOW_GEMV_MM` to 16 to satisfy the static assert, so a B=32 blob would give every sequence
  from 16 up a zero logit row. `plowc` happily wrote `gv_mm_max: 32` into `build.json`.
  `AmdEngine::load` now refuses it; the emit side still does not.
- **The register-cliff gate cannot fail above MM=4**: the allocator clamps at 256 and converts
  overflow into scratch **spill**, so MM=8/16 pass `total > 256 || occ < 2` while spilling
  19–20. Above MM=4 the binding signal is spill, not total. The gate does not check spill.
  (Unchanged from 2026-07-27.)
- **Paged KV and a runtime-variable batch are the next work**, per §6a — not a decode
  micro-optimisation. The cache being a compile-time `[B][...]` allocation is what caps B at
  8 and what makes a wide blob cost 1.9x latency at concurrency 1.
- **TP is still batch 1.** `AmdTpGroup::submit_decode` takes one scalar `(pos, kvlen)`, so a
  batched TP packet would collapse every rank onto sequence 0. `AmdServe::load` refuses the
  combination rather than serve a wrong token. GLM-5.2 TP4 therefore gains nothing here yet.
- **No chunked or interleaved prefill.** One prefill occupies the whole device and stalls
  every other slot for its duration, so TTFT under load is bounded by the longest prompt in
  flight, not by a chunk. This is what drives the TTFT column in §6, and it is the next lever.
