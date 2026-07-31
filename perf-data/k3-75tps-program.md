# Reaching 75 tok/s on K3 fp8 — the arithmetic first, then a ranked experiment program

**Goal:** decode 75 tok/s, K3 fp8, plus servable `plowrt` with chunked prefill, prefill+decode,
GSM8K and a speed bench.

Everything below is against **33.24 ms/token = 30.1 tok/s**, measured 2026-07-30 on 8× gfx950,
K3 93 layers TP8, ctx 32000, `--steps 200`, UNBOUND, under `perf-data/harness/gpulease` with
`foreign-during=0`. 75 tok/s is **13.33 ms/token — a 2.49× speedup.**

---

## 0. THE GATING FACT: neither half alone gets there

`kimi-k3-README.md` §7 splits the token ~20 ms protocol / ~19 ms bodies. Scaled to the 33.24 ms
measured here that is **protocol ~17.0 ms | bodies ~16.2 ms**, and the body term has a hard
bandwidth floor of **2.837 ms** (14.021 GB/rank/token at 6200 GB/s), i.e. bodies run at **18% of
peak**.

```
protocol -> 0        leaves 16.2 ms = 62 tok/s     SHORT
bodies   -> floor    leaves 17.0 ms = 59 tok/s     SHORT
```

**Every single-lever plan fails.** A split that lands on 13.33 ms:

| protocol | bodies | speedup needed | bodies as % of peak BW |
|--:|--:|---|--:|
| 5.0 ms | 8.3 ms | 3.4× / 2.0× | 34% |
| **4.0 ms** | **9.3 ms** | **4.3× / 1.7×** | **31%** |
| 3.0 ms | 10.3 ms | 5.7× / 1.6× | 28% |

So the program is: **protocol ~4×, bodies ~1.7×.** Anything that does not move one of those two
numbers is not on the path, however good it looks.

**And note what this says about batch 1 vs throughput.** 75 tok/s *per stream* needs the 2.49×
above. 75 tok/s *aggregate* is a different and much easier target — the roofline crossover on
MI355X is ~batch 312, so below that decode is HBM-bound and aggregate throughput scales nearly
linearly with concurrency. **Speculative decoding is the only lever in this document that raises
per-stream tok/s without shortening the chain**, which is why it belongs in the per-stream column
and continuous batching does not. Decide which number the goal means before spending on §4.

---

## 1. Where we already are, measured

| arm | ms/token | tok/s | status |
|---|--:|--:|---|
| baseline | 33.24 | 30.1 | shipping |
| + `PLOW_L2_PLACE` | 32.89 | 30.4 | numerically neutral, needs a bound gate |
| + HIER2 **ceiling** | 28.44 | 35.2 | **UNSAFE** — data race, prices the sound version |
| + both | 28.15 | 35.5 | ditto |

The HIER2 ceiling is the single largest measured lever and it is **−14.7%**. Landing it soundly
takes 33.24 → ~28.5, i.e. **35 tok/s — under half way**. That is the honest headline: the biggest
thing on the board gets less than a fifth of the distance.

---

## 2. PROTOCOL — needs ~4× (17.0 → 4.0 ms)

| # | experiment | mechanism | expected | how to price cheaply | state |
|---|---|---|--:|---|---|
| **P1** | **Sound HIER2** | two XCD-local rendezvous on `nper[p][d]`; one `buffer_wbl2`+`buffer_inv` per XCD instead of per workgroup | **−4.9 ms** | already priced by `PLOW_GATE_HIER_CEIL` | **building** |
| P2 | `PLOW_L2_PLACE` default | per-XCD queues; also what makes `nper[p][d]` static | −0.2 ms, and P1 depends on it | done; needs bound token gate | ready |
| P3 | K3 dispatch-width audit | GEMVs dispatched wider than their waves can feed; protocol cost scales with workgroup-packets | **~−1.8 ms** est. | `plowrt disasm` + the emitter's own saturation rule, no GPU | **never done on K3** |
| P4 | Chain depth | 1739-deep critical path × per-packet cost is a floor no per-packet win escapes | bounds everything | `--counters` disasm, no GPU | analysed, unattacked |
| P5 | CPX partition mode | on a 1-L2 config the ISA defines `buffer_wbl2 sc1` as a **NOP** — the cost is removed by construction, not reduced | protocol → ~0 | `ctr_convergence.hip` under CPX vs SPX, one lease, no model | **untested, highest info/cost** |

**P3 is the cheapest unclaimed win and it is CPU-only.** GLM got −3.5 ms from exactly this audit
(`xrfit` −1.82, rope-fit −1.148, combine-fit −0.506) and K3 has never had it. My earlier pass over
the K3 blob found ~50,700 workgroup-packets of wave-underfed width across nine GEMV shapes; at the
36.2 ns/workgroup-packet that the GLM A/Bs calibrate, that is ~1.8 ms.

**P5 is the only item that changes the ceiling rather than approaching it.** It needs a
machine-wide `amd-smi` partition change, so it must be scheduled when the box is free.

---

## 3. BODIES — needs ~1.7× (16.2 → 9.3 ms, 18% → 31% of peak)

Two ways only: move fewer bytes, or move them faster.

| # | experiment | mechanism | expected | state |
|---|---|---|--:|---|
| **B1** | **`PLOW_K3_SHARD_HEAD`** | column-parallel `lm_head`; the README puts it at **−14.7% GEMV bytes** | **~−2.4 ms** | **BUILT, gate passes**, blocked only on an equivalence check that tolerates a `vocab/tp`-wide logits dump |
| B2 | bf16 → fp8 audit | GLM's analogous audit found **13.36 of 19.20 GB** was still bf16 (MLA projections, `o_proj`, shared expert, `lm_head`). K3 is mxfp4-expert + fp8-KV but its projections are unaudited | large if it mirrors GLM | **unaudited on K3** |
| B3 | expert width / EP | GLM's `MoeExpertDown` ran at 12% of peak because TP made K=512; EP restores K to the full expert width. K3 at TP8 has `imoe/8` — same shape | ≤−2 ms by analogy | analysed on GLM only |
| B4 | GEMV achieved bandwidth | `lm_head` hits **94.1% of peak** on the same kernel that does 23% elsewhere, so the kernel is fine and the shapes are not | the residual | shape-by-shape |

**B1 first: it is built, measured, gated, and worth more than any protocol item except P1.** The
only thing between it and shipping is that `k3_tp_equivalence.sh` cannot check it, because
`--dump-logits` dumps `act.logits` which is `vocab/tp` wide at tp=8 and its shape check fails by
construction. That is a harness fix, not a kernel one.

**B4's control is the important one to keep quoting:** the same `d_gemv` reaches 94.1% of peak on
`lm_head` and 23% on the projections. Nothing in the body half is a kernel-quality problem; it is
a shape and byte problem.

---

## 4. THROUGHPUT AND SERVING — a different metric, and mostly different work

| # | item | why | metric it moves |
|---|---|---|---|
| T1 | continuous / in-flight batching | 255 of 256 CUs are idle on a latency-bound chain at batch 1 | **aggregate** only |
| T2 | chunked prefill | bounds TTFT and lets prefill interleave with decode | TTFT, and aggregate |
| T3 | **speculative decoding** | accepts k tokens per chain traversal — the ONLY lever that divides the 1739-deep chain per *emitted* token | **per-stream** |
| T4 | paged KV | gates how many sequences fit; a contiguous `max_ctx` allocation per sequence caps concurrency hard | aggregate |

T3 deserves its own arithmetic: at an accept rate of `a` tokens/step it multiplies per-stream
tok/s by `a` for a step cost that grows sub-linearly (the chain is traversed once, the GEMVs get
wider — and *wider is free here*, because at batch 1 they are at 18% of peak). **A 2–3× accept
rate on a 28.5 ms post-HIER2 token is 70–105 tok/s per stream.** That is the single most likely
route to the goal as stated, and it is cheaper than finding 1.7× in the body half.

**Prefill is already 2.25–2.79× better** as of the `k3-mla-prefill-mfma` merge (one MLA layer, fp8
latent, n_head=12: ctx 32768 141.6 → 50.7 ms; a 32k prompt's four chunks 324.7 → 120.1 ms/layer,
7.8 → 2.9 s per rank over 24 MLA layers). The README's "24 s to first token at 32k" is stale by
that factor and should be re-measured before any further prefill work is scoped.

---

## 5. ORDER OF WORK

1. **P1 sound HIER2** — in flight; the largest measured lever, and P2 rides along.
2. **B1 `PLOW_K3_SHARD_HEAD`** — built and idle; needs only a harness fix. Best ratio on the board.
3. **P3 K3 dispatch-width audit** — CPU-only, no GPU, ~1.8 ms.
4. **Re-measure prefill/TTFT** post-mfma-merge, so §4 is scoped against the truth.
5. **B2 bf16 audit** — one emit + the sidecar's declared bytes; tells us if GLM's 70%-bf16 finding
   transfers.
6. **T3 spec-decode feasibility** — the goal as stated is probably unreachable without it.
7. **P5 CPX price check** — one lease, no model, and it is the only item that could make §2 moot.

## 6. METHOD RULES THIS CAMPAIGN LEARNED THE HARD WAY

* **`perf-data/harness/gpulease`, never a bare `flock /tmp/plow_gpu.lock`.** The bare lock
  serialises but neither waits for nor warns about a concurrent agent's campaign. Timed that way
  one arm read **46.418 ms at position 2 and 33.802 at position 1** — the sign followed the order.
* **UNBOUND for sub-ms work**, bound only for correctness: a bound run re-reads 195 GiB/rank and
  the second of any pair is penalised 5–10 ms.
* **Interleave and reverse the order every rep**, and quote the control's own spread.
* **Any edit under `runtime/amd/*` stales every tuning record** — `tuned_tile_selection` is the
  signal, `rebench_tune_gemm_all.sh --bf16-only` is the fix. It fired twice in one day here.
* **Any new kernel must pass `scripts/sc1_coverage.sh`.** It has now caught unscoped stores in two
  separate incoming branches, in kernels that did not exist when the audit ran.
* **Price a ceiling with a deliberately-wrong knob before building the sound thing.** Every
  conclusion in this document that cost nothing came from that discipline; every one that cost a
  day came from skipping it.

---

# 7. CORRECTIONS FROM THE SERVING AUDIT — three of §4's assumptions were wrong

A read-only audit of the serving/batching path (`plowrt serve`, `mux.rs`, `serve/engine.rs`,
`exec/amd.rs`, the bench scripts) corrects this document in three places. All three matter.

## 7.1 Speculative decoding is NOT the cheap route to per-stream 75 tok/s

§4 called T3 "the single most likely route to the goal". **It is blocked on K3 specifically**, and
not by scheduler work:

* `crates/plowrt/src/orch/speculative.rs` is **151 lines of unwired host arithmetic** (34 of them
  tests). `SpeculativeWorkflow`, `SpecConfig` and `accepted_prefix` have **zero callers** anywhere
  in the tree. There is no draft model, no Medusa head, no EAGLE, no n-gram predictor.
* The decode program's `t` is **compiled, not passed** (`serve/engine.rs:99`), and the sequence
  axis and the token axis are the same axis. "Verify k drafts in one dispatch" is not a shape the
  blob can currently express.
* The device **samples argmax on-device and the host never sees the logit row**
  (`mux.rs:1258-1268`) — so top_p/top_k/penalties are already ignored on this backend, and a
  draft-verify needs logits the host cannot currently obtain.
* **And the disqualifying one:** rejecting a speculated suffix requires rolling the state back.
  KV is append-only and rewinding `pos` works, but **K3's KDA recurrent state is
  read-modify-written in place with no snapshot** (`exec/amd.rs:3264-3276`). 69 of 93 layers are
  KDA. There is nothing to roll back to.

## 7.2 …but that blocker is the SAME blocker as batching, and that is the useful finding

K3 decode is **structurally batch-1**, refused twice at load:

```
exec/amd.rs:3271  "KDA state has no batch axis, so the per-slot stride below would alias every
                   sequence's state onto every other's. Batched decode over a recurrent state
                   needs a paged-state indirection (slot table + pool) that does not exist yet"
serve/engine.rs:187-195  TP x batch refused: AmdTpGroup::submit_decode is still scalar
```

**A slot-indexed, snapshottable KDA state pool unblocks concurrency AND speculative rollback with
one piece of work.** That is the highest-leverage item on the whole board, because it is the only
thing that opens both the aggregate axis and the per-stream axis at once. It pairs with paged KV,
which `batched-decode-amd-status.md:184` already names *"the single highest-value follow-up"* —
the flat `[B][kv_head][ring][hd]` allocation is 22.5 GiB at B=8/16k and 45 GiB at B=16, so B is
bounded by VRAM at the WORST-CASE context rather than by live demand.

## 7.3 Prefill: the MFMA win covers 24 of 93 layers, and TTFT is unmeasured

The merged `d_flash_mla_prefill_mfma` is 2.25–2.79× on **MLA** — 24 layers. The other **69 layers
are KDA, and KDA chunked prefill is NOT DONE** (`kimi-k3-kernel-gap.md:847` lists it NEW/L:
*"Until then TTFT is prefill-as-N-decodes"*). So:

* the ~24 s TTFT-at-32k figure is the ONLY number in the tree, it **predates** the MFMA fix, and
  no post-fix end-to-end K3 TTFT has been measured;
* the MLA component went 7.8 → 2.9 s/rank, but the KDA majority is untouched and is
  prefill-as-N-decodes;
* **prefill is not protocol-bound** — the T=8192 program's critical path is 2109 packets ≈ 12 ms,
  **0.2% of a 24 s TTFT**. It is bodies, and it is KDA.

**Measure the post-merge TTFT before scoping anything else in prefill.**

## 7.4 What already works, so it does not need building

* **`plowrt serve` WORKS and K3 loads** — all three historical blockers closed (`abc019f`,
  `abb7af0`, `429b01f`). `/v1/chat/completions` + `/v1/models` + `/healthz` + `/metrics`, SSE
  streaming. No `--hsaco`/`--checkpoint` flags: `<assets>/hsaco`, `<assets>/checkpoint`. Requires
  `--features hsa` or it silently serves the CPU reference through a byte-fallback tokenizer.
* **Chunked prefill WORKS** as a bucket-ladder DP (`plan_chunks`, `exec/amd.rs:1071-1085`) — a
  1500-token prompt becomes 1024+512, not two padded 1024s. What is ABSENT on gfx950 is
  **chunk↔decode interleaving** (`PLOW_PF_INTERLEAVE`/`PLOW_PF_CHUNK`/`PLOW_PF_DEFER_DECODE` are
  all `#[cfg(feature="cuda")]`), so one prefill holds the whole device for a tick.
* **A serving harness exists**: `scripts/bench_plowrt_serve.sh` drives `vllm bench serve
  --backend openai-chat`, and it already has a coherence gate ("a fast wrong server is not a
  result") and a rejected-request trap (`gen_toks` must equal `num_prompts × OUTLEN`).

## 7.5 Do not build: disaggregation

`scripts/disagg_phase0.sh` priced it before building, and the answer is NO-GO. Peak prefill share
is **28.6%** (conc 16) against a **50%** go/no-go threshold — below that, a 1:1 prefill:decode
split over N GPUs is a throughput LOSS versus N co-located replicas. `orch/disagg.rs` is a 23-line
stub with no callers; leave it there.

## 7.6 Accuracy: ABSENT

No GSM8K, no lm-eval, no MMLU anywhere in the tree. What exists is **token-identity** gating
(`k3_tp_equivalence.sh`, the Paris continuation gate, the serve coherence gate) — which proves
self-consistency, not quality. A throughput number without an accuracy number is not publishable
against vLLM, so the harness has to be built.

---

# 8. REVISED ORDER OF WORK

| # | item | unblocks | cost |
|---|---|---|---|
| 1 | **P1 sound HIER2** (in flight) | −4.9 ms decode | in progress |
| 2 | **B1 `PLOW_K3_SHARD_HEAD`** | −2.05 GB/rank/token = **−14.7% GEMV bytes** | harness fix only |
| 3 | **Measure post-MFMA TTFT** | scopes all prefill work; current number is stale | one lease |
| 4 | **P3 K3 dispatch-width audit** | ~−1.8 ms | CPU only |
| 5 | **Slot-indexed + snapshottable KDA state** | **batching AND speculative decode** | large, highest leverage |
| 6 | **Paged KV** | B beyond 8–16 | large, pairs with 5 |
| 7 | **GSM8K harness** | makes any throughput number publishable | small |
| 8 | KDA chunked prefill | TTFT on 69 of 93 layers | large |

Items 2, 3, 4 and 7 are all small and none of them are blocked. Items 5 and 6 are one project and
they are what a serving benchmark actually needs.

---

# 9. B1 LANDED, AND IT CORRECTS HOW THE WHOLE BODY HALF MUST BE RANKED

## 9.1 The gate the README calls impossible is the wrong gate

`kimi-k3-README.md:152` records `PLOW_K3_SHARD_HEAD` as ungateable: *"`k3_tp_equivalence.sh`
CANNOT gate it — `--dump-logits` dumps `act.logits`, `vocab/tp` wide at tp=8, so its shape check
fails by construction."* That is true and it does not matter: **token identity does not need the
logits dump at all.** Each rank argmaxes its own vocab slice and `XArgmaxFin` reduces to the same
global argmax, so the emitted TOKEN is the object to compare.

Measured, K3 93 layers TP8, real weights, `--ctx 5`, prompt `1008,10484,318,15383,387`:

```
control     [13, 646, 12259, 387, 14868, 220, 5807, 6017, 1873, 13, 646, 7695, ...]
shard_head  [13, 646, 12259, 387, 14868, 220, 5807, 6017, 1873, 13, 646, 7695, ...]   IDENTICAL
```

Prefill → 17374, all 8 ranks agree, 32/32 ids identical. The knob is verified: `lm_head`'s GEMV
goes `N=163840 → N=20480` in the blob, exactly `vocab/tp`.

*(Caveat kept honest: this is one prompt at one ctx. A default flip deserves the multi-prompt
form, but the mechanism — sharded argmax then cross-rank reduce — is now demonstrated sound.)*

## 9.2 The gain is REAL and it is 6× smaller than the byte count implies

| arm | ms/token | sd |
|---|--:|--:|
| base | 33.160 | 0.060 |
| `PLOW_K3_SHARD_HEAD=1` | **32.781** | 0.096 |

**−0.379 ms, −1.14%**, every one of 4 interleaved pairs favouring it. §3 predicted **−2.4 ms** from
"−14.7% GEMV bytes". That prediction was wrong, and the reason is the whole lesson:

```
2.055 GB removed, of 14.021 = 14.7% of GEMV bytes
  at PEAK 6200 GB/s ..................... 0.331 ms
  at the 18% the body AVERAGE runs at ... 1.841 ms
  MEASURED .............................. 0.379 ms   <- peak, not average
```

**`lm_head` was already running at ~94% of peak** — it is the very control this campaign quotes to
prove `d_gemv` is not the problem (94.1% on `lm_head` against 23% on the projections). Its bytes
were therefore the *cheapest bytes in the model*, and deleting 14.7% of the bytes deleted 1.14% of
the time.

## 9.3 THE RE-RANKING RULE FOR EVERY REMAINING BODY LEVER

> **A byte-reduction lever is worth `bytes_removed / achieved_bandwidth_of_those_bytes`, not
> `bytes_removed / peak`. Cutting bytes that already move at 94% of peak buys ~4× less per byte
> than cutting bytes that move at 23%.**

This inverts §3's ordering. The body half's money is **not** in the tensors with the most bytes —
it is in the tensors with the worst achieved bandwidth:

* **B4 (achieved bandwidth on the projections, 23% of peak) moves from "the residual" to FIRST.**
  The same GB moved at 23% instead of 94% costs 4.1× the time, so fixing the *rate* on the
  6.355 GB of "everything else" is worth far more than removing bytes from anything already fast.
* **B2 (bf16 → fp8) must be re-costed per tensor**, weighted by each tensor's current efficiency,
  not by its size. GLM's 13.36-of-19.20-GB bf16 finding is a byte count, not a time estimate.
* **B3 (EP for expert width)** rises, because its whole mechanism IS an efficiency fix: GLM's
  `MoeExpertDown` ran at 12% of peak *because* TP made K=512, and EP restores the rate.

The instrument this needs is the per-opcode `tok GB/s` / `%peak` column that
`glm52-decode-attribution.md` produces, run on K3. **That attribution has never been done on K3**
and it is now the highest-value CPU+one-lease item in the body half, because without it every
byte-reduction estimate in §3 is unweighted and therefore wrong in the same way §9.2 was.

---

# 10. THE ATTRIBUTION EXISTS NOW, AND IT OVERTURNS §3 AND §9 BOTH

`scripts/k3_rate_attrib.py` (new) is the K3 twin of `glm52_token_attrib.py`'s `%6200` column.
`k3_trace_report.py` had **zero** byte accounting, so this had never been measured on K3.

Traced, K3 TP8 ctx 32000, leased, 2459 packets, body 28.726 ms, 14.021 GB/rank/token:

| opcode | pkts | ms | %body | GB | GB/s | %peak |
|---|--:|--:|--:|--:|--:|--:|
| **GEMV** | 816 | **8.660** | 30.1% | 14.021 | 1619 | **26.1%** |
| XREDUCE | 278 | 4.278 | 14.9% | — | — | — |
| FLASH_MLA_DECODE_FP8 | 24 | 2.926 | 10.2% | — | — | — |
| ATTN_RES | 187 | 2.534 | 8.8% | — | — | — |
| MOE_ROUTER_TOPK | 92 | 2.063 | 7.2% | — | — | — |
| GEMV_QKVG | 69 | 1.821 | 6.3% | — | — | — |
| GEMV_GLU | 92 | 1.124 | 3.9% | 1.013 | 901 | 14.5% |

Read naively that says **7.2 ms is recoverable by rate** (GEMV 8.66 → 2.41 at the 94% `lm_head`
demonstrably reaches with the same kernel). **That reading is wrong, and this is the finding.**

## 10.1 Per shape, against the per-packet floor

The empty b=256 packet costs **5.72 µs** (real trace). Compare each shape's *bytes* against it:

| op | N | K | cnt | µs/pkt | MB | µs for the bytes @94% | bound by |
|---|--:|--:|--:|--:|--:|--:|---|
| Gemv | 163840 | 7168 | 1 | 381.3 | 2348.8 | 403.0 | **BW — 99.4% of peak** |
| Gemv | 3584 | 7168 | 92 | 15.9 | 51.4 | 8.8 | **BW** |
| Gemv | 896 | 7168 | 92 | 12.2 | 12.8 | **2.2** | PROTO |
| GemvGlu | 768 | 7168 | 92 | 12.2 | 11.0 | **1.9** | PROTO |
| Gemv | 896 | 3584 | 92 | 11.5 | 6.4 | **1.1** | PROTO |
| Gemv | 7168 | 1536 | 93 | 11.0 | 22.0 | **3.8** | PROTO |
| Gemv | 1536 | 128 | 69 | 7.7 | 0.4 | **0.1** | PROTO |
| Gemv | 12 | 7168 | 69 | 6.2 | 0.2 | **0.0** | PROTO |

**14 of 17 GEMV shapes move bytes that take LESS TIME THAN AN EMPTY PACKET COSTS.** Their low
`%peak` is not a bandwidth failure — it is the packet floor showing through. Rewriting the kernel
for them would change nothing.

```
Bandwidth-recoverable (shapes whose bytes dominate the floor):   0.683 ms
Protocol residue above the 5.72 us floor (HIER2's territory,
  ALREADY COUNTED — not additive):                               3.235 ms
```

## 10.2 What this means

**§3's premise is wrong. The body half is not a bandwidth problem — it is the protocol again,
wearing a `%peak` disguise.** Corrections:

* **"bodies need 1.7×" is not achievable by kernel or byte work.** Only **0.683 ms** of the GEMV
  term is bandwidth-recoverable. B2/B3/B4 as written in §3 are chasing a number that is not there.
* **§9's re-ranking rule was right but its conclusion was not.** "Cut the worst-rate tensors"
  fails when the worst rate is caused by the packet floor rather than by the kernel. The rule
  needs a second clause: **a byte lever only pays where the bytes already dominate the packet
  floor** — which on K3 decode is `lm_head` (now sharded, and it duly paid only 0.379 ms) and the
  one 51 MB expert-down shape.
* **The 2.837 ms bandwidth floor is unreachable at batch 1 by construction.** 1739 packets on the
  critical path × 5.72 µs = 9.95 ms of protocol before a byte is read. The model cannot approach
  its own roofline because its packets are too small to amortise the protocol.

## 10.3 Which makes BATCHING the structural answer, not a throughput side-quest

In a protocol-bound regime, batch `B` divides **both** terms per emitted token: the same weight
bytes serve `B` sequences, and the same packet floor is amortised over `B` tokens. That is exactly
why the MI355X roofline crossover is ~batch 312 — below it, decode is not bandwidth-bound at all.

So the ordering inverts once more, and this time it agrees with §7.2 rather than fighting it:

1. **Protocol per packet** — HIER2 (−4.9 ms measured), then whatever follows it.
2. **Chain depth** — 1739 × floor is the hard term; fusion has failed repeatedly, so this needs a
   different idea, not another attempt.
3. **Batching** — the only lever that divides a protocol-bound token, and it is blocked on the
   single KDA-state gap that also blocks speculative decode (§7.2).
4. **Bandwidth work** — worth **0.683 ms**. Do it last, or not at all.

**75 tok/s = 13.33 ms against a 9.95 ms protocol chain floor at today's per-packet cost.** At
HIER2's 3.46 µs the floor is 6.02 ms and the target becomes arithmetically reachable — but only
with the packet cost fixed first. Nothing in the body half gets there on its own.
