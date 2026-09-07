# CPU backend — pending plan

Written 2026-09-07. Baseline: `origin/main` at `dfea70f` (PR #19 merged), plus branch
`cpu-vllm-26b-baseline` (3 perf-data commits, no code).

Goal being chased: beat llama.cpp **and** vLLM by a margin on Gemma-4 and GPT-OSS-20B, all data
types, everything measured through `plowrt serve` on the OpenAI API.

## 0. Where the scoreboard actually stands

| model / data type | vs llama.cpp | vs vLLM | status |
|---|---|---|---|
| Gemma-4-12B bf16 | 1.15x | 2.0x | **done** |
| Gemma-4-12B fp8 | 1.05x | 3.6x | **done** |
| Gemma-4-12B MXFP4 | 1.49x | 5.7x | **done** |
| Gemma-4-26B-A4B MXFP4 | wins all measured | **no baseline obtainable** | closed, see §5 |
| GPT-OSS-20B MXFP4 | 32/32 cells | **26/32 cells** | 6 cells open, see §1-§2 |

The 6 open GPT-OSS cells split into exactly two causes:

* **4 TPOT cells at c>=2** — not kernel speed. Batched MoE decode at rung 8 runs a step in 100 ms
  against vLLM's 137 ms measured TPOT, i.e. we are 1.37x *faster per step*; the served 142-195 ms
  means 42-95 ms per token is spent waiting behind another request's prompt. Needs prefill+decode
  fusion. **User-reserved — do not start.**
* **2 long-prompt TTFT cells** — pure prefill speed, no interference component. Sized in §1.

Re-measure the full 32-cell matrix before trusting the 26/32 split: it predates the epilogue
transpose, the L2 token chunking, the M-split, and `PLOW_MX4_HEAD`.

## 1. P0 — MoE prefill kernel speed (in flight, blocking the 2 TTFT cells)

### The sizing, measured 2026-09-07

Per-op profile, GPT-OSS-20B, 1024-token prefill, `--threads 16`
(`perf-data/tools/gptoss-prefill-profile-1024.sh`):

| op | busy/thr | share of 1491.8 ms mean busy |
|---|---|---|
| `MOE_GLU_MX_PF` | 614.2 ms | 41% |
| `MOE_DOWN_MX_PF` | 335.1 ms | 22% |
| `FLASH_PREFILL` | 248.1 ms | 17% |
| `GEMM` | 235.0 ms | 16% |

Efficiency against exact MAC counts (1024 tok x 24 layers x top-4 x 3 mats x 2880^2 = 2.446e12
MACs; GLU 2/3, DOWN 1/3):

| op | per physical core | vs 1464 GMAC/s achievable TMUL |
|---|---|---|
| GLU | 332 GMAC/s | **22.7%** |
| DOWN | 304 GMAC/s | **20.8%** |

Not bandwidth (9.55 GB of expert weights per 1024-token chunk is ~95 ms at ~100 GB/s, against
949 ms of MoE time). Not dequant (~19%, already hoisted out of the token-block loop).

**Target: MoE prefill ~44% faster** (949 -> ~519 ms/thr) closes served summarize c=1 from 2238 ms
to ~1829 ms. Anything less narrows but does not flip the cell.

### Status

An agent is mid-pass on `x_moe_glu_mx_pf` / `x_moe_down_mx_pf` in
`runtime/cpu/dev/amx/moe_amx.c`. Current state of its work:

* `v1` — preserves output, **performance-neutral** (617.6 vs 614.2 ms/thr, inside noise). Tells us
  the limiter is not where the first hypothesis put it.
* `v1d5` / `v1d6` — reports GLU 143.7 and DOWN 66.7 ms/thr, prefill wall 2024.5 -> 923.2 ms, i.e.
  4.3-5.0x. **Not bit-exact**: the profiler's `tokens:` line goes from `first 976 next 3206 "TheIt"`
  to `first 290 next 2543 " the line"`. Treated as unverified — a 4-5x against a 44% target,
  arriving with changed output, is far more likely to be dropped work than a legal reordering.

### Gate before any of this lands

1. `tokens:` line identical to baseline on the 1024-token profile.
2. A shape-pinned golden test in `runtime/tests/cpu_dev_amx_test.c` at the shapes GPT-OSS prefill
   actually uses — I=2880, K=2880, E=32, top-4, **and a non-multiple-of-32 row count**. ctest
   passing today is necessary but not sufficient: if no existing shape reaches the path, wrong
   output still passes. This is the same class of hole that "three GEMM shapes that actually reach
   the M-split (no existing shape did)" closed on the last branch.
3. ctest 12/12, plowrt `--features cpu` and devgen suites green.
4. Interleaved serve A/B on summarize c=1, 2-3 pairs, same session (box drifts ~5%).
5. Full 32-cell GPT-OSS matrix plus the Gemma-4-12B three-type ladder, to catch a regression
   elsewhere — the MoE prefill ops are shared with the 26B.

If the speed genuinely requires a numeric change: keep the default bit-exact, gate the fast arm
like `PLOW_MOE_INT8`, and record the accuracy cost. A gated 4x is valuable; a silently wrong
default is not.

### Specific failure modes to rule out on the fast variant

* Fewer than `clen` real rows computed, or a mishandled partially-filled token block (row-count
  fields are rewritten in `kvrow.rs`; partial buckets must compute only `clen` rows).
* Skipped experts or token-expert pairs — the total must stay 1024 x top-4 per layer.
* Dropped K tail when K is not a multiple of 32, or `nxb`/scratch left unset so a block no-ops.
* Stale or uninitialised C tile / accumulator read instead of the dequantized strip.

## 2. P1 — scheduling: 26% of prefill wall is idle (bit-exact, untouched)

Same profile: wall **2024.5 ms** against **1491.8 ms** mean busy. Worker min 1377.8, max 1587.5.

* Perfect balance alone floors wall at max-busy **1587.5 ms — a 21.5% cut**, which on its own is
  larger than the 18.3% the TTFT cell needs.
* Decompose: ~210 ms is imbalance (min-to-max spread), the remaining ~437 ms is dependency stalls
  on the counter-gated packet DAG.
* Note this 26% is at 16 workers on 8 physical cores. The 15% on record was at a narrower width —
  SMT contention shows up here, and TMUL is shared per physical core.

### MEASURED 2026-09-07 11:0x — width is worth -18.5%, bit-exact

The 8-vs-16 worker experiment has now been run, and it is the largest bit-exact result on the
table. All rows below have a `tokens:` line identical to baseline:

| variant | threads | prefill wall | GLU ms/thr | DOWN ms/thr | delta |
|---|---|---|---|---|---|
| baseline | 16 | 2024.5 | 614.21 | 335.08 | — |
| v2 (asm) | 16 | 1953.0 | 594.46 | 332.97 | -3.5% |
| v3 (asm) | 16 | 1912.4 | 585.25 | 325.99 | -5.5% |
| **v2t8** | **8** | **1650.0** | **566.10** | **310.32** | **-18.5%** |

Scaled to the served cell: 2238 x 1650/2024.5 = **~1824 ms against vLLM's 1829**. So width alone
plausibly flips summarize c=1, with **no numeric change at all** — where the asm restructuring is
worth only 3.5-5.5%. This inverts the priority in §1: the kernel pass is the bonus, width is the
lever.

Two caveats before believing it: that is a profile wall, not a measured TTFT, and the run set
`threads=8` for the **whole engine**. A global change would regress batch-1 dense decode, which
wants logical CPUs. The correct shape is per-phase width.

Work items, in order of expected value:

1. **Per-phase worker width — now P0.** Task #15 landed per-*model* width (physical for MoE,
   logical for dense) but per-phase needs idle workers to stop polling first; check whether that
   blocker is still real. Needed: a proper width sweep (8/10/12/16, bit-exact, wall + per-op
   busy/thr — only 8 and 16 exist so far), confirmation that the win is SMT contention on the
   shared TMUL rather than scheduling overhead (worker idle was 26% at 16 threads; what is it at
   8?), and proof the decode step does not regress at the chosen width.
2. **Attack the ~437 ms of dependency stalls.** Look at whether MoE prefill serialises behind
   `MOE_ROUTER_TOPK_PF` / `MOE_ALIGN_PF` (0.63 and 0.03 ms/thr busy, but spans of ~1932 ms — they
   sit on the critical path across the whole prefill). Check per-CU static stream assignment.
3. **Balance the non-MoE ops.** Expert slice partitioning is already weighted by rows-per-expert;
   `FLASH_PREFILL` and `GEMM` partitioning has not been checked for the same skew.

All bit-exact by construction — packing workers and reordering independent packets changes no
accumulation order.

## 3. P2 — prefill+decode fusion (BLOCKED, user-reserved)

Do not start. Recorded sketch, for whoever designs it: one program with
`T = prefill_chunk + n_decode` rows, dense and MoE ops over all rows unchanged, `FLASH_PREFILL` on
`[0, clen)` and `FLASH_DECODE` on `[clen, clen + n_decode)`. 8 decode rows inside a 1024-row
prefill cost ~13 ms against ~105 ms standalone.

Already measured and dead as a substitute: `--pf-interleave 512` moved chat_long c=4/c=8 TPOT from
100/195 to 100/198 — the work is throughput-bound, not stall-bound.

## 4. P3 — lower-value / not started

| item | evidence | verdict |
|---|---|---|
| fp8 KV cache (#13) | measured ~4% of decode traffic | low value, deprioritise |
| int8 w8a8 dense weights | emitter flags exist, no CPU kernels at any tier | not started |
| `PLOW_MOE_INT8` default-on | -9.1% on summarize TTFT for activation-int8 error | needs a quality call, see §7 |
| `pack_x_panel` M-split | ~1.2-1.5% of prefill wall, bit-exact | small; may already be covered by 0e3d2ec |

## 5. Gemma-4-26B vs vLLM — closed, with the mechanism

Not obtainable on this box, and not for the reason previously recorded. vLLM 0.28.0 CPU **has**
four x86 quantized MoE expert paths (fp8, MXFP4, int4, int8), all AMX-gated, and this box has AMX.
Every one requires a SILU-family activation; Gemma-4's MoE is GELU-tanh (`gemma4.py:368`). The only
GELU-capable x86 path is unquantized, pinning experts at bf16: 47.00 GiB of text weights against
58.85 GiB of RAM, hence the engine-core death at init.

Consequence for planning: **a bigger box does not fix this.** It would let vLLM run the model in
bf16 while still refusing every quantized CPU expert kernel. So any future 26B comparison is
plow-MXFP4 vs vLLM-bf16 — a legitimate comparison, but a different framing that should be stated
rather than presented as like-for-like. Full trace: `perf-data/cpu-gemma26b/vllm-baseline.md`.

## 6. Scale-out to 64/128 cores — pending, unmeasured

Nothing here is measured; this box is 8c/16t. What will bind, in likely order:

1. **Parallelism granularity.** GPT-OSS prefill has 32 experts and ~128 token-rows per expert at
   1024 tokens. At 64+ workers the per-slice work approaches one 32-row AMX block, so the weighted
   expert partition stops balancing. Needs a second partition axis (K or N within an expert).
2. **TMUL sharing.** One TMUL per physical core, shared by SMT siblings — the physical-vs-logical
   worker width rule was derived at 8 cores and must be re-derived, not assumed.
3. **NUMA.** Multi-socket puts expert weights across nodes. `PLOW_L2_PLACE` exists and the original
   design contemplated GQ domain cursors as NUMA queues; neither is validated. Weight placement
   must be node-local or the 9.55 GB/chunk expert read crosses the interconnect.
4. **The 26% idle from §2 will get worse before it gets better** — more workers on the same
   counter-gated DAG means more waiters on the same critical path.
5. **L2 budget.** `WM_L2_BUDGET` is 1536 KiB against this part's 2 MiB private L2. Per-core L2
   differs across parts; this should read the actual cache size, not a constant.

Do §2 before §6: fixing balance at 8 cores is a prerequisite for reasoning about 64.

## 7. Decisions needed (blocking, cannot be resolved by measurement)

1. **Fusion: go or no-go.** Blocks 4 of the 6 open GPT-OSS cells. Everything else in this plan is
   already unblocked.
2. **Policy on non-bit-exact defaults for TTFT.** `PLOW_MX4_PREFILL` already ships on with
   documented divergence. `PLOW_MOE_INT8` is -9.1% for activation-int8 error and is off. If the §1
   agent's fast arm turns out to be legitimately faster but lossy, the same question decides
   whether it ships on. A consistent rule is better than a per-flag judgement each time.
3. **26B framing** — accept plow-MXFP4 vs vLLM-bf16 as the comparison, or record the cell as
   "no valid baseline" permanently?

## 8. Operational notes that have cost real time

* **Interleave every A/B.** This box drifts ~5% over hours; the same blob read 2469 ms at 22:25 and
  2603 at 06:53. Nothing below ~10% is trustworthy across sessions.
* **Use the mean, and check which statistic a recorded number is.** The "~27% more prefill
  throughput" figure in the record came from a 2603 ms reading that was a **p50**; the bench table
  column is `TTFT mean/p50/p90`. Today's paired means are 2238 (bit-exact) and 2034 (int8).
* **Know whether a flag is emit-time or runtime.** `PLOW_MX4_PREFILL` and `PLOW_MX4_HEAD` are
  emit-time — toggling them against a stale blob produces a flat, meaningless A/B. `PLOW_MOE_INT8`
  is runtime, so it A/Bs on one blob.
* **One model process at a time**, quiet gate, and never kill a process another session started.
* **Never `pgrep -f` a pattern that matches your own command line** — it self-matches, and an
  `until ! pgrep -f ...` loop then never exits.
* **Write long runs to a log file**; piping one through `tail` got it OOM-killed with all output
  lost.
* **Disk is at ~97%** (8.9 GiB free). Blob dirs are small (2.4 MB — weights live in
  `PLOW_MXFP4_DIR`), but a full re-quantize needs room. Check before emitting.
* **Re-emit every blob after an opcode renumber.** A merge once gave three opcodes two meanings.
