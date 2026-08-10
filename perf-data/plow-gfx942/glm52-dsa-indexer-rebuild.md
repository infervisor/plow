# The DSA prefill indexer, priced and rebuilt

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4), one rank · **CDNA3-CRITICAL** — the closing argument DEPENDS on fp8 MFMA running at bf16 rate, which is true on CDNA3 and FALSE on CDNA4 (2x). The indexer arithmetic floor is arch-independent; the conclusion that fp8 cannot shrink it is CDNA3-only.

Base `worktree-glm52-bringup` @ 9b06b05, branch `dsa-indexer`. Directive: the sparse-prefill
counterfactual in `glm52-current-cost-decomposition.md` §4 is eaten by plow's OWN selection chain,
so attack the indexer, not the flash. Everything below is measured on gfx942 (MI300X), one rank,
GLM-5.2 TP8 indexer geometry (`index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048).

## 0. What the indexer actually is, in FLOPs — the number that bounds every answer here

Per causal (query, key) pair the lightning indexer computes `Σ_h w[t][h]·ReLU(q[t][h]·k[s])` over
32 heads of 128 dims = **8,192 FLOP/pair**. The dense MLA flash it is supposed to replace costs
9,216 (QK over 8 local heads × 576) + 8,192 (PV over 8 × 512) = **17,408 FLOP/pair**.

**The indexer is 47% of the dense flash's arithmetic, and it runs over the SAME causal pair set.**
That is a floor, not an implementation defect, and it is the single most important fact for the
net calculation: selection can never be free, and on CDNA3 it cannot even be made cheaper by
dropping precision — this box's fp8 MFMA runs at bf16 rate (`glm52-experiments.md (consolidated)`), so vLLM's
fp8 indexer buys bandwidth here and zero arithmetic.

So the question is not "can the indexer be made negligible" (it cannot) but "how far above its own
arithmetic floor is plow's implementation". That is what §1 answers.

## 1. Where the cost actually is — from the ISA, not the source

Ops 117/118 were lifted out of the megakernel into standalone kernels
(`runtime/amd/test_kernels.hip`) so they could be disassembled and timed in isolation;
`runtime/bench/interp/dsa_pf_indexer_bench.c` + `scripts/build_dsa_pf_indexer_bench.sh` are the
instrument. Every arm is resolved **by name out of the loaded code object**, so there is no
possibility of measuring an arm the run never opened.

### 1.1 op 117 `IndexScorePf` is VMEM-ISSUE bound on operand re-fetch, not matrix-unit bound

The shipped work item is one (query, 32-key) subtile, and it fetches BOTH MFMA operands from
global every time: 8 KiB of query head-rows + 8 KiB of key rows to produce 32 scores. The source
comment argues this is fine because "keys are L2-resident" — but L2 residency does not fix an
instruction-issue limit. Disassembled (`llvm-objdump --mcpu=gfx942`), one 32-key subtile is:

| | shipped `index_score_pf_128` | rebuilt `index_score_pf_row_128` |
|---|---:|---:|
| `v_mfma_f32_32x32x8_bf16` | 16 | 16 |
| `global_load_dwordx4` in the MFMA loop | 16 | **0** |
| `global_load_dwordx2` (the `W` weights) | 4 | **0** (hoisted to VGPRs) |
| `ds_read_b128` | 0 | 8 |
| `s_waitcnt vmcnt(...)` in the loop | 12 (several `vmcnt(0)` = full drains) | **0** |
| `s_and_saveexec_b64`/`s_or_b64 exec` pairs | 8 | **0** |

The shipped inner loop issues ~60 non-MFMA instructions per 16 MFMA and hard-serialises on
`s_waitcnt vmcnt(0)` between them; the `if (pos < len)` key guard is re-evaluated INSIDE the
k-loop as eight exec-mask save/restore pairs. The rebuilt loop is MFMA plus LDS reads and waits
only on `lgkmcnt`, which is an order of magnitude cheaper and software-pipelined (`lgkmcnt(1)`
lets one `ds_read` overlap the next MFMA pair).

At T=8192 over a 16384 KV that re-fetch is 3.15e6 work items × 16 KiB = **51 GB of cache traffic
per layer** to do 825 GFLOP of MFMA.

### 1.2 The T×ctx f32 matrix is real, but it is the SMALLER half

The matrix everyone (including this campaign's own notes) has blamed is 403 MB of causal-live
entries at T=8192/ctx=16384 — about 0.13 ms at HBM rate. Op 118 then re-reads it **eight times**
(seven MSB-first radix passes plus the emit pass) = 3.2 GB, ~1.1 ms. Both are real and both are
attacked below, but neither is the 51 GB.

**So the prime suspect named in the directive — "does the score matrix need to exist at all" — is
NOT where the cost is.** Materialisation costs ~0.13 ms/layer; operand re-fetch costs an order of
magnitude more. Fusing score into select (never writing the matrix) would have bought the smaller
term and would have COST occupancy, because a fused kernel must hold a whole row.

## 2. What vLLM actually does (checked, not assumed)

The reference the directive names is worth stating precisely, because it is not what the campaign
assumed. In `/workspace/vllm-rocm-build`, the prefill indexer **does materialise a dense fp32
`[M_chunk, N_chunk]` logits tile in HBM** and runs top-k as a separate kernel; scoring and
selection are NOT fused. Query-axis chunking exists to bound the tile under a 512 MB budget
(`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`), not to avoid the traffic. Two things there DO transfer:

1. **The gfx942 scoring kernel is already row-resident.** AITER's Triton `fp8_mqa_logits` launches
   one workgroup per query row with `BLOCK_KV=128` and the 64 heads on the MFMA M axis — the
   operands are staged and reused, exactly the decomposition §3 adopts. plow's op 117 is the
   outlier, not vLLM's.
2. **Their top-k stages candidates.** `topKPerRowPrefill` histograms into 2048 bins, emits
   strictly-better elements directly, stages the threshold bin into ≤2048 shared-memory slots and
   early-exits. plow's op 118 re-reads the entire row on every one of its seven passes and stages
   nothing.

The fp8 half does not transfer: on CDNA3 it is a bandwidth trick with no arithmetic payoff.

## 3. The rebuild

Both arms are **exact**, and the bench gates that rather than asserting it.

**op 117 arm B — `d_index_score_pf_row`** (`runtime/amd/op_attention.h`). Work item becomes
(pack of 8 query rows, span of KV positions); wave *w* owns query `p*8+w`, so the whole workgroup
shares one key stream. The query's A-fragments and its 16 lane-local lightning weights are hoisted
into VGPRs once per work item; keys stream contiguously through LDS one `TILE_N` slab at a time.
Key traffic falls ~8× (one read per pack instead of per query-subtile), query traffic ~`SPAN/32`×.
Same 8 k-steps in the same order into the same accumulator, same w-weighted ReLU epilogue, same
scale ⇒ **bit-identical output**, gated byte-for-byte over the whole causal matrix.

`TILE_N` is the occupancy knob and is measured, not assumed: 128 keys is 34,816 B of LDS (1
workgroup/CU, 2 waves/SIMD), 64 keys is 17,408 B (3 workgroups/CU, 6 waves/SIMD).

**op 118 arm B — `FAST_EXIT`.** `d_index_select_coop` (the decode selector) has a measured
fewer-passes early-out that `d_index_select_pf` was written without: `dsa_pack_key_a` puts the
whole 32-bit score in the top four bytes, so after pass 3 the score threshold is exact and the
remaining three passes exist only to split a genuine tie by index. When the boundary bin's
population equals the still-needed count the tied group is wholly selected and those passes cannot
change the emitted set. That removes 3 of the 8 full-row scans. Selection is unchanged — gated by
top-k **set** equality per row (emit order is arbitrary by design).

### Landing discipline

Both arms are wired into `interp.hip` behind `PLOW_DSA_IDX_ROW` (**default ON**, `=0` restores the
shipped arms for an A/B). They are default-on rather than opt-in because they are bit-identical
and strictly cheaper, and the sparse-prefill path they serve is itself default-OFF — no shipped
program moves. The LDS slab is bounded by a `static_assert` against the interpreter arena, because
hand-sizing that arena is precisely how the MoE-PF prefill once shipped a live LDS overrun.

The arm carries a symbol marker `plow_dsa_idx_row_arm` (verified present in a real gfx942 object
next to `plow_mla_pf_ns_arm`) so "which arm is in this object" is answerable by symbol scan. It
deliberately carries **no `requires` entry**: the refuse-at-load machinery exists for arms that
change what a packet's operands MEAN, and this one changes no operand, no packet field and no
output value. A pre-arm object given the same packet runs the shipped kernel and is correct, just
slower — degrade, not corrupt.

## 4. Measured — gfx942, 304 workgroups, 9 reps, median, isolated kernels

Spread is `(max−min)/median` over the 9 reps. Every cell passed its gate: the score arms are
**byte-identical over the whole causal matrix**, the select arms agree on **2048/2048 positions on
every row checked** (64 rows per config).

| config | op 117 shipped | 117 row TN=128 | 117 row TN=64 | op 118 shipped | 118 FAST | chain shipped → rebuilt |
|---|---:|---:|---:|---:|---:|---:|
| T=4096 ctx=4096 | 1129 ±15% | **318** (3.55×) | 321 (3.52×) | 634 ±3% | **361** (1.76×) | 1.763 → **0.678 ms** |
| T=8192 ctx=8192 | 3689 ±4% | **1262** (2.92×) | 1279 (2.88×) | 1827 ±3% | **981** (1.86×) | 5.516 → **2.243 ms** |
| T=8192 ctx=16384 | 8800 ±4% | **2500** (3.52×) | 2568 (3.43×) | 3048 ±1% | **1729** (1.76×) | 11.848 → **4.229 ms** |

µs per layer. The speedups are 2–3 orders of magnitude above the control spread.

**Two caveats on what these numbers are contingent on.** op 117's timing is data-INDEPENDENT (no
branch depends on a score), so its 2.9–3.6× is unconditional. op 118's is not: `FAST_EXIT` fires
only when the boundary bin's population equals the still-needed count, i.e. when no exact-score
tie has to be split by index. On the bench's synthetic f32 scores that is always, so 1.76–1.86×
is the *upper* bound — on a distribution with real boundary ties the arm degrades gracefully to
the shipped 7 passes (1.0×), never to something wrong. The selection is exact either way. Second,
the select gate samples 64 rows per config (192 rows total), not every row; at T=4096/ctx=4096
about half of those take the `row_len <= top_k` identity path, so the radix path is exercised by
the rows above 2048 and by every row of the two larger configs.

**TN=128 and TN=64 are within 2% of each other**, which independently confirms the diagnosis: if
the kernel were latency-bound the 3×-higher occupancy of TN=64 (6 waves/SIMD vs 2) would have won
clearly. It was issue-bound on operand re-fetch, and once the re-fetch is gone occupancy is
irrelevant. TN=64 is what ships (17,408 B — it fits the interpreter LDS arena with room).

**Rate achieved by op 117** (indexer FLOPs = 8,192 per causal pair):

| config | GFLOP/layer | shipped | rebuilt |
|---|---:|---:|---:|
| T=4096 ctx=4096 | 68.7 | 60.9 TF/s | 216.2 TF/s |
| T=8192 ctx=8192 | 274.9 | 74.5 TF/s | 217.8 TF/s |
| T=8192 ctx=16384 | 824.7 | 93.7 TF/s | **329.9 TF/s** |

The shipped arm ran at 5–7% of the MI300X bf16 MFMA peak. The rebuild reaches 25%, which is
better than the dense MLA flash itself achieves on this box (§4's traced flash span at the same
shape works out to ~203 TF/s). Further headroom exists — the loop is now LDS-read bound, and two
queries per wave would halve `ds_read` per MFMA — but §5 shows it would not change any decision.

### Reconciliation with the trace-derived figures (important)

`glm52-dsa-sparse-b2.md` recorded op 117 at "3.7 ms/layer at 16k"; isolated, it is 8.8 ms/layer at
T=8192/ctx=16384. Two reasons, and neither is a contradiction: (a) that figure priced a 16k prompt
as if it were one chunk, but `plan_chunks` covers 16k as **[8192, 8192]** — two chunks with very
different causal pair counts, so the whole-prompt cost is their SUM; (b) prefill packet spans
OVERLAP (`glm52-current-cost-decomposition.md` §0), so a traced span attributes less than a
kernel's exclusive cost. The numbers below use the **isolated** basis, which is the conservative
direction: it makes the indexer look expensive and therefore makes sparse look worse.

## 5. The net calculation, redone — and it does NOT clear the bar

Whole prompt, ×78 layers, chunk-decomposed. Flash columns are §4's; the indexer column is
measured here. (32k is extrapolated at the µs/pair rate of the largest measured config.)

| T | flash dense | ideal sparse | gross saving | %TTFT | indexer SHIPPED | NET | indexer REBUILT | **NET** | **%TTFT** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 77 | 63 | 14 | 1.4% | 138 | −124 | **53** | **−39** | **−4.0%** |
| 8192 | 240 | 118 | 122 | 7.3% | 430 | −308 | **175** | **−53** | **−3.2%** |
| 16384 | 915 | 249 | 666 | 18.4% | 1354 | −688 | **505** | **+161** | **+4.4%** |
| 32768 | 3570 | 511 | 3059 | — | 5051 | −1992 | **1824** | **+1235** | — |

**Answer to the directive's question 3, stated plainly: no.** The rebuild is real — it removes 62%
of the selection chain and it is exact — but sparse DSA prefill still **does not clear the ≥15%
bar at 8k or 16k**. At 8k it is still NEGATIVE (−3.2%). At 16k it is +4.4%, a third of the bar.
Note also that §4's original table was optimistic for a reason unrelated to kernel quality: it
priced the whole prompt from a single chunk and counted op 117 only, so its "+378 ms / +10.4% at
16k" overstated the shipped position too.

### And it cannot be fixed by making the indexer better

The indexer computes `Σ_h w·ReLU(q_h·k)` over 32 heads × 128 dims = **8,192 FLOP per causal pair**,
against the dense MLA flash's 17,408. **Selection is intrinsically 47% of the arithmetic of the
attention it is trying to avoid**, over the same pair set, and on CDNA3 fp8 cannot shrink it (fp8
MFMA runs at bf16 rate here). So there is a hard floor. Priced at 100% of the bf16 MFMA peak for
scoring and full HBM rate for selection:

| T | perfect indexer | NET | %TTFT |
|---:|---:|---:|---:|
| 8192 | 37 ms | +85 | 5.0% |
| 16384 | 149 ms | +517 | **14.2%** |

**A perfect indexer — one that cannot be built — still lands under 15% at 16k.** The route is not
kernel-limited; it is limited by the arithmetic of selection itself. This closes indexer
optimisation as a way to make sparse pay at ≤16k, in the same audit sense that
`glm52-dsa-sparse-b3.md` closed the union route.

### The one lever that would clear it, and it is not a kernel

The indexer is **replicated on all 8 TP ranks** (all 32 index heads on every rank), while the flash
is sharded 8 ways (8 of 64 attention heads per rank). That replication is why the per-rank ratio is
47% rather than 6%. Sharding the indexer's **query axis** across ranks and all-gathering the
selected indices (not the scores — the scores are a T×ctx matrix and un-gatherable) gives:

| T | indexer compute /8 | + all-gather of `iidx` | total | NET | %TTFT |
|---:|---:|---:|---:|---:|---:|
| 8192 | 22 | 19 | 41 | +81 | 4.8% |
| 16384 | 63 | 38 | 101 | **+565** | **15.6%** |

(all-gather priced at `T×2048×4 B×7/8` per layer over plow's measured 240 GB/s; u16 positions
would halve it.) That is the first configuration in this campaign where sparse crosses the bar at
16k — and it is an **emit + collective** change, not a kernel one. It is also the only remaining
one: it composes with this rebuild, and nothing else on the table moves 47% to 6%.

**Caveats that must travel with that row.** The all-gather is a new serial dependency in a layer
that already spends 12% of TTFT in collectives, so the byte cost is a floor, not an estimate; and
+15.6% is at the bar, not over it. It should be scoped before it is built, not built on this row.

## 6. Verdict

1. **The indexer was 3× off its own achievable rate, and that is fixed, exactly.** op 117 −2.9…3.6×
   bit-identical, op 118 −1.8× set-identical, chain −62%. Landed default-on behind
   `PLOW_DSA_IDX_ROW`, marker `plow_dsa_idx_row_arm`, LDS bound asserted.
2. **The prime suspect was wrong.** The T×ctx f32 matrix costs ~0.13 ms/layer to write; the
   operand re-fetch cost 51 GB/layer. Never materialising the matrix — the thing the directive
   asked about first, and the thing vLLM turns out not to do either — was the smaller half.
3. **Sparse still does not clear the bar at 8k or 16k, and a perfect indexer would not either.**
   Do NOT build the per-query membership-skipping flash on this evidence: its own quantified
   prize (§4's `ideal sparse`) is already counted above, and the total still comes to +4.4% at 16k
   measured / +14.2% ideal.
4. The only route that crosses the bar at 16k is TP-sharding the indexer's query axis (+15.6%).
   That is the thing to scope next, and it is an emitter change.

Nothing here changes any shipped program: the sparse path remains default-OFF on both axes, and
these arms only execute inside it.
