# GLM-5.2 DSA: the degenerate output, found and fixed (2026-07-28, gfx950 / ROCm 7.2.4)

The code for everything below landed in **3699ff1**, which is a *dispatch-width* commit — a
concurrent agent's `git commit -a` in the shared worktree swept these files up before they could be
committed under their own message. The commit title therefore does not describe them. This file is
the record. Files involved: `runtime/amd/op_attention.h`, `runtime/amd/interp.hip`,
`runtime/amd/test_kernels.hip`, `runtime/bench/dsa_gather_bench.c`, `crates/devgen/src/mla.rs`.

Context: knob-contract §6g-DSA. plow ran DENSE where vLLM runs SPARSE because `PLOW_GLM_DSA=0` was
forced on every long-context blob and `final_bench.sh` pinned `GLM_CTX=65536`, both because the DSA
path was recorded as producing **degenerate output**.

## 1. The one-token `atomicExch` hypothesis is FALSIFIED. Do not re-litigate it.

The suspicion: `op_attention.h`'s

```c
if (bid == 0 && tid == 0) Cg[2] = 0u;   /* reset emit slot */
```

is a plain global store, where the histogram reset three lines below uses `atomicExch`, and the
function's own contract says communication is "exclusively through L2-coherent ATOMICS". On 8-XCD
gfx950 a plain store could land dirty in one XCD's L2 while every other workgroup's
`atomicAdd(&Cg[2],1u)` is performed at a coherent point — which would make every layer after the
first select nothing (`slot >= top_k` always) and silently re-attend the **previous layer's rows**.
`act.igctl` is shared across all 78 layers and host-zeroed once, so the reset is load-bearing on
every layer after the first. The candidate fix was one token: `atomicExch(&Cg[2], 0u)`.

**It is not the bug.** Two independent measurements:

* **ISA.** Disassembling `index_select_coop_a` (ROCm 7.2.4, gfx950): the reset is
  `global_store_dword v1, v1, s[10:11] offset:8` and the emit-slot bump is
  `global_atomic_add v1, v5, v1, s[10:11] offset:8 sc0`. **`sc0` is the return-previous-value bit,
  not a scope bit, and neither instruction carries `sc1`** — both operate in the same
  hardware-coherent L2 domain, so the store is observed.
* **End to end, at HEAD, unmodified.** `dsa_gather_bench`'s tie-stress re-launches the selector over
  a *different* score array sharing one `gCtl` and reports the selected set **EXACT** at ctx
  8k / 32k / 128k / 256k. A missed reset would have left the previous launch's selection in `idx[]`
  and the check would have failed.

A comment at the line now pins this so the next reader does not spend an afternoon on it.

## 2. The actual bug: the selector was never told the live KV length

`d_index_score_mfma` writes `Score[pos]` only `if (pos < kv_len[b])`. But the emitter baked
`i[0] = ctx` — the packet's **max** ctx — into `INDEX_SELECT`, and nothing patches it per decode step
(`derive_kvrow` has no `IndexSelect` arm, and none is wanted). So the radix ranked `ctx - kv_len`
words **the score kernel never wrote**. DSA arms only above a 64k crossover, so that was essentially
the whole array on every real decode step.

Two ways it goes wrong, both deterministic — not a race, which is why the symptom reproduced:

1. the indexer score is `Σ_h w[h]·ReLU(q·k)·scale` and `weights_proj` emits negatives, so genuine
   scores go **negative** and an untouched `0.0` hole **outranks** them;
2. with `kv_len < top_k` (any short prompt) fewer than 2048 rows exist at all, so the difference was
   padded straight out of the uninitialised tail.

Either way the gather received positions past the end of the cache — and
`d_flash_mla_decode<…,GATHER=true>` applies **no mask** (`keep` is just `kv < hi`; the set is
"assumed causal because the selector produced it") — so it read latent rows that had never been
written. That is the recorded "degenerate output even on the decode-only bundle".

`runtime/nvidia/op_dsa.cuh` already records this class of defect **against this AMD kernel**, tagged
`[RAG]`: *"The AMD kernel assumes len > top_k … If len <= top_k it never satisfies k_rem, leaves
idx[len..top_k) UNWRITTEN, and the gather then reads uninitialised indices out of bounds."* It fixed
its own half with an explicit `n_sel` count. It missed the deeper half: on AMD `len` was `max_ctx`,
so even `len > top_k` ranked unwritten scores.

### Fix

`kv_len` is now an operand of the selector (`t[4]`), which clamps `len = min(i[0], kv_len)` and
`top_k = min(i[1], len)`. The gather derives the **same** `min(top_k, len)` from the **same** operand,
so the two agree by construction rather than by a matching pair of emit-time constants — no `n_sel`
tensor is needed. `top_k` still strides `ibase` (that is the table's allocation stride, which does
not shrink with the live length). Empty splits were already safe: both `d_flash_merge` and
`d_mla_merge_fold` guard on `-inf` / `gl > 0`.

## 3. Oracle: before vs after, identical harness and data

`dsa_gather_bench` with a new **short-context gate**. It poisons the whole score buffer with `+1e30`,
then lets the real score kernel overwrite only `[0, kv_len)` — so the surviving tail is exactly what
production leaves there, reproduced rather than simulated. "Before" is the same tree with **only the
two clamps neutralised**, so the ABI and everything else match.

| kv_len / max_ctx | expected idx | before | after |
|---|--:|---|---|
| 2049 / 8192 | 2048 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 777 / 8192 | 777 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 8192 / 32768 | 2048 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 777 / 32768 | 777 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 32768 / 131072 | 2048 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 777 / 131072 | 777 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 65536 / 262144 | 2048 | MISMATCH — **index out of range** | **EXACT**, gather PASS |
| 777 / 262144 | 777 | MISMATCH — **index out of range** | **EXACT**, gather PASS |

Two regimes are covered deliberately: `kv_len > top_k` (the ordinary long-prompt case) and
`kv_len < top_k` (a short prompt, where fewer than `top_k` rows exist).

**The pre-existing `kv_len == ctx` gate reports EXACT in BOTH columns.** That is why this survived so
long: the bench uploaded `kv_len = ctx` and passed `len = ctx`, so score-written and select-scanned
lengths were equal *by construction*. The blind spot was structural, not a gap in coverage.

An emit-side guard was added too
(`glm_dsa_selector_is_bound_to_the_live_kv_length_and_declares_its_geometry`): the ops were all
present and correctly ordered the whole time, so only a test that inspects packet **fields** can see
this. devgen lib suite: **179/179 pass**.

## 4. Register delta: ZERO

Measured with `build_gfx950.sh`'s own cliff check, in isolation against HEAD 9dc27bb:

| bucket | HEAD | +DSA fix |
|---|---|---|
| prefill | 256 VGPR / 0 AGPR / occ 2 / spill 2 | **identical** |
| decode | 254 VGPR / 0 AGPR / occ 2 / spill 0 | **identical** |
| flash | 256+256 / occ 1 / spill 228 | **identical** |

Composes freely with the concurrent GF=8 flash arm and the gemv walk.

## 5. Two silent-emit holes closed alongside

Both refuse at **emit time**. This follows the tree's existing doctrine (`op_moe.h`,
`moe_bound_topk`): the AMD interpreter's dispatch `default:` is a deliberate silent NOP, so the
compile/emit refusal is the only loud failure available. Adding the first device-side `__trap()` to
the megakernel is the kernel owner's call, not a backstop's.

* **Indexer geometry.** `cfg_glm` parsed `index_n_heads` / `index_head_dim` from config.json with no
  validation, while `interp.hip` hardcoded `DI_ = 128, HI_ = 32` and `d_index_score_mfma` carries
  `static_assert(HIc == 32)` — HI=32 is baked into the 32×32 MFMA accumulator fragment layout, not a
  template knob. A checkpoint with a different geometry parsed cleanly, sized `qidx` as `hi*di`, and
  was then strided by the kernel as 32*128. `GlmCfg::dsa` now refuses anything else, and the packet
  carries `i[1]`/`i[3]` (the ISA contract in `dev_isa.h:419`, which the emitter had left at zero) so
  it is self-describing. Real GLM-5.2 is `index_n_heads=32, index_head_dim=128, index_topk=2048` —
  the shipping model is unaffected. The `kimi_ref_cfg` fixture that armed DSA at `index_heads: 8`
  was the reachable case and is corrected.
* **`attention_head_dim` (Nemotron).** `exec_flash_decode`, `FLASH_MERGE` and `HEADNORM_ROPE` are
  if-chains over {128, 256, 512} with **no `else` and no trap**, so `hd = 64` or `192` selects
  nothing: Q never RoPE'd, `Opart` never written, `O` never written. The only prior shape gate was
  `hd % 8 == 0`, which both pass. `cfg_nemotron` now refuses at the config.json boundary — chosen
  over an emit-time assert because the synthetic `nemo_ref_cfg` fixture uses `attn_head_dim: 16` for
  op-sequence tests that never build a runnable object.

## 6. What the data implies for `CROSSOVER` — and what is still PROJECTED

`CROSSOVER = 65536` (`mla.rs`) is badly stale. Measured per-layer attention, `dsa_gather_bench`,
tp8 head shard (NH=8), ns16:

| ctx | dense (µs) | gather (µs) | score (µs) | select (µs) |
|---|--:|--:|--:|--:|
| 8k | 114.5 | 98.2 | 16.8 | 53.3 |
| 32k | 260.4 | 99.2 | 17.5 | 53.6 |
| 128k | 894.4 | 98.1 | 21.2 | 63.6 |
| 256k | 1793.0 | 98.2 | 28.5 | 79.1 |

Gather is **flat at ~98 µs** (top_k=2048, exactly as designed); dense fits `46.7 + 0.006623·ctx` µs.
Real GLM-5.2 has `indexer_types` = **21 full / 57 shared** over 78 layers, so the indexer amortises
to ~22.5 µs/layer:

| ctx | dense | DSA (gather + amortised indexer) | ratio |
|---|--:|--:|--:|
| 8k | 114.5 | 117.1 | 0.98x (dense) |
| 32k | 260.4 | 118.3 | **2.20x DSA** |
| 128k | 894.4 | 120.9 | **7.40x DSA** |
| 256k | 1793.0 | 127.2 | **14.1x DSA** |

**Measured microbench crossover ≈ 11.2k — the shipped constant overestimates it by ~5.9x.**

### Measured vs projected — read this before quoting any of it

* **MEASURED:** everything in §1–§5, and the µs table above. All of it is op-level or
  compile-level.
* **PROJECTED:** the ~11.2k crossover. It is an extrapolation from op-level microbenches whose
  widths differ (the selector is benched on a 32-WG slice, dense/gather on the full chip), and it
  ignores everything outside the attention chain.
* **NOT MEASURED AT ALL:** TTFT and TPOT. No end-to-end number is claimed here. That needs a full
  78-layer blob under `plowrt` through `vllm bench serve` (§0-BENCH), plus model-level oracle
  agreement on real weights — this work establishes correctness at the *kernel* level only.

**`CROSSOVER` is deliberately NOT changed.** The MoE confound makes that mandatory: on GLM-5.2 any
arm with wrong numerics **over-reports**, because routing is data-dependent and garbage activations
collapse the router's top-k so the expert ops do less work. A DSA timing without model-level oracle
agreement at the same config is inadmissible. Lowering the constant to ~8–16k is a hypothesis this
data supports; it needs an 8-GPU sweep behind it.

## 7. Loose ends for whoever picks this up

* **The end-to-end validation is still owed**: a 78-layer GLM-5.2 blob emitted with DSA armed, run
  under `plowrt` against `runtime/tests/glm52_real_oracle.py`, then TTFT/TPOT at 4k/16k/64k/128k
  DSA-on vs the dense baseline. Not done here — the box had two live `plowrt` jobs on GPUs 4–7 and
  an armed benchmark waiter.
* `d_index_select_coop`'s `red` parameter is documented `[2]` but the `FAST` path writes `red[2]`
  (a third element). Harmless — the interp carves it from the 147 KB raw arena and the test wrappers
  declare `red[4]` — but the comment was wrong and is now `[3]`.
* `d_index_select_coop` has **no batch loop**: it writes `ib[slot]` with no batch offset, while
  `d_index_score_mfma` scores `b < n_batch`. Fine today (every decode packet emits `n_batch = 1`),
  but it is an unstated precondition, not a property.
* The `[RAG]` note in `runtime/nvidia/op_dsa.cuh` should be updated: the AMD half it describes is now
  fixed, and by a different mechanism (shared `kv_len` operand rather than an `n_sel` output).

## 8. THE END-TO-END VALIDATION CAME BACK **NEGATIVE**. Measured 2026-07-29, gfx950, TP4.

Section 7 owed an end-to-end run. It has now been done, and **the DSA arm still produces degenerate
output on the real 78-layer GLM-5.2 checkpoint at every context length tested.** The kernel-level
result above stands exactly as written — it is just not sufficient. **No DSA TTFT or TPOT number
exists, `CROSSOVER` must stay at 65536, and `PLOW_GLM_DSA=0` remains the only servable setting.**

Bundles: `glm52_tpctx_sweep.sh emitdsa`, TP4, `--max-ctx 135168`, one tree (HEAD = `0bba733`), one
plowc, one object directory (`build-amd/dsa-objs`, built from HEAD; registers byte-identical to the
table in §5). The two arms differ in **one emit-time bit**, `PLOW_GLM_DSA`. Unlike the GF pair,
`build.json` *does* separate them: the DSA arm carries `FlashGatherDecode` + `IndexScore` +
`IndexSelect`, the dense arm `FlashMlaDecode`.

Needle-in-a-haystack probe (`longcoherence`), greedy, `max_tokens=24`, identical prompts on both
arms (both calibrated to the same 15.03 tok/repeat):

| arm | prompt tokens | result |
|---|--:|---|
| **dense**, published bundle | 118,792 | **PASS** at depth 0.1 / 0.5 / 0.9 — `CRIMSON-FALCON-10/50/90` |
| **dense**, HEAD emit + fresh objects | 19 / 20 / 2,116 | **PASS** — Paris, `17*23 = 391`, Tokyo, streaming carries content |
| **DSA** | 118,792 | **FAIL** — `'CR\n\n</think></think></think>…'` |
| **DSA** | 8,017 | **FAIL** — `'CR\nComments\nComments\nComments…'` |
| **DSA** | 2,032 | **FAIL** — identical degeneracy |
| **DSA** | 532 | **FAIL** — identical degeneracy |

Three things this pins down, none of which need another lease:

1. **It is not the object build and not the emitter.** The dense bundle emitted from the *same*
   tree and run on the *same* fresh objects is fully coherent. The DSA arm is the only difference.
2. **It is not a long-context bug.** It fails identically at 532 prompt tokens.
3. **IT IS NOT THE SELECTION.** At `kv_len = 532` the fix clamps `top_k = min(2048, 532) = 532`, so
   the selector emits **every** live row and the gather attends to all of them. Softmax is
   permutation-invariant, so a DSA step at `kv_len < top_k` is *mathematically identical to dense*
   no matter how bad the scores are. It is not. So the defect is **downstream of, or beside, the
   ranking** — in the gather's operands or in what the extra indexer ops do to shared state — and
   the `kv_len` fix in §1–§4, while necessary and correct, was not the last one.

**THE STRONGEST CANDIDATE, from the emitter, GPU-free: the indexer key cache is never filled by
prefill.** `kv.{l}.kidx` (`[ctx][DI]`, allocated per *full* indexer layer at `mla.rs:1098`) is
written at exactly one site — `mla.rs:1927`, inside the **decode** DSA block, one row per decode
step at the current slot. `grep kidx` finds **no write in the prefill emitter and no occurrence at
all in `crates/plowrt/` or `runtime/amd/`**. So after a P-token prefill, rows `0..P-1` of the
indexer key cache hold whatever the arena held, and `d_index_score_mfma` scores every prompt
position against uninitialised memory. Read as f32 that is readily NaN, and a NaN in the packed-key
radix does not yield a permutation even when `top_k == len` — which is precisely the one case
observation 3 says must otherwise have been safe. Consistent with the whole ladder, and consistent
with the first emitted token (`CR`) being correct: **that token comes from the dense prefill; every
token after it comes from a DSA decode step.**

This is §4's bug shape once more, one level up from the fix in §3: the score kernel's *input cache*
has no producer on the prefill path, and `dsa_gather_bench` cannot see it because the bench uploads
a fully-populated key cache. Whoever picks this up should start by asserting, on device, that
`iscore[0..kv_len)` is finite on the first decode step after a prefill.

**Two probe defects fixed in `scripts/glm52_tpctx_sweep.sh` on the way in** — both would have made
this verdict unreadable, and both were latent because `longcoherence` had never been run:

* **No backend gate.** Its first run came up in 5 s on `hsa_init failed: 4104`, silently selected
  the **CPU reference backend**, answered `''` at every depth and printed a confident `FAIL`. (Root
  cause of the 4104: this account is not in the `render` group — every GPU run on this box has to go
  through `sg render -c`, which the script header documents and which is easy to drop.) The probe
  now refuses to run if the server log says `CPU reference backend active`.
* **The filler was 2.2x oversized.** The constant `N = 17000` was annotated "~7 tokens per repeat";
  the filler is **15.03** tokens per repeat, so it built a **255,033**-token prompt against a
  135,168-token blob — past `max_ctx`, i.e. the KV-ring overrun class this repo has hit three times,
  and it would have read as a model failure. `N` is now calibrated against the server's own
  tokenizer with one cheap 200-repeat request, and `NEEDLE_TOKENS` takes a *list* so one server load
  can walk a ctx ladder.

## 9. THE REPRO IS NOW 15 SECONDS, NOT 4 MINUTES. `scripts/glm52_dsa_l4_oracle.sh`.

§8 was found the expensive way: every iteration paid the full 78-layer, 183 GiB/rank weight load
first — 167 s at TP4, 255 s at TP8 — which `runtime/tests/glm52_decode.c:224` already calls "the
whole cost of a run". That was the wrong vehicle and it is recorded here so the next person does not
repeat it. **Full-network loads burned on iteration in this session: 4** (one TP4 DSA needle probe,
one DSA ctx ladder, and two aborted starts), roughly 35 minutes of lease time to learn something a
truncated model shows in seconds.

`GLM_NLAYERS=N` (`mla.rs:3569`) truncates the emit to the first N layers **while keeping the full
serving structure and the TP degree**. It must be used *with* `GLM_FULL=1` — the single-layer gate
(`GLM_FULL` unset) asserts `tp == 1` at `mla.rs:3860` and cannot do TP4/TP8. Measured here:

| bundle | layers | server ready |
|---|--:|--:|
| `tp4` (published) | 78 | **167 s** |
| `l4-dense-tp4` | 4 | **12 s** |
| `l4-dsa-tp4` | 4 | **15 s** |

**THE GATE AT 4 LAYERS IS AN ORACLE, NOT COHERENCE.** A 4-layer truncation of a 78-layer model emits
gibberish by construction, so "does it say Paris" is meaningless. What is meaningful is the identity
§8 turns on: **at `kv_len <= top_k` a correct DSA step is arithmetically dense.** The selector clamps
`top_k = min(index_topk, kv_len)` and therefore emits *every* live row; the gather attends to all of
them; softmax is permutation-invariant. So for any prompt shorter than `index_topk` (2048) the DSA
arm and the dense arm **must emit the same tokens, gibberish or not**. No reference implementation is
needed, and the check is three short prompts.

**Result, TP4, 4 layers, greedy, `max_tokens=32`, three prompts — FAIL, and it reproduces §8 exactly:**

```
dense  ' Sequenzatcp compsRTC_synthetic tensazenbiaoszcartoonserpQUEenchbumptechuwndeza...'
dsa    ' Sequenza Heights器和 outlettensisman1交前任 Mikhailssueottideaux4 Compact愧rz...'
```

The **first token is identical** (` Sequenza`) and every token after it diverges — the same signature
as the full model's `'CR'` followed by `</think>` spam. The first token is produced by the **dense
prefill**; every later token is a **DSA decode step**. All three prompts behave this way, and all
three are ~20 tokens, i.e. two orders of magnitude below `top_k`, where the arms are provably equal.

Iterate here. Only `longcoherence` and the published TTFT/TPOT table need the full 78 layers, because
those are dominated by prefill over the whole stack and a truncation would measure a different thing.

## 10. THE END-TO-END BUG IS FOUND: THE TWO INDEXER KERNELS STRIDE ON `blockIdx.x`, NOT ON `slice`

**§8's "strongest candidate" (`kv.{l}.kidx` has no prefill producer) is a REAL defect and it is NOT
this failure.** The kidx claim survives verification (§10.3) but it cannot explain a divergence at
`kv_len = 532`, and §8's own observation 3 is the proof: at `kv_len <= top_k` the clamped selector
emits EVERY live row *whatever the scores are* — including NaN. Work the radix through: with
`top_k == len` the boundary scan drives `prefix` down to the MINIMUM key on every pass, `#{key >=
prefix} == len`, and the emit loop writes all `len` positions. Garbage keys change the ORDER, and
softmax does not care about order. So the defect had to be somewhere the score values do not reach.

(§8 also over-charged the NaN half of its own argument: `d_index_score_mfma`'s epilogue is
`part += wlds[h] * (d > 0.0f ? d : 0.0f)`, and `NaN > 0.0f` is FALSE — the ReLU **sanitises NaN to
zero**. A garbage key cannot NaN-poison the score array through that reduction.)

### 10.1 The bug, and why the op-level oracle could never see it

`runtime/amd/op_attention.h` had **exactly two** `blockIdx.x` references in the whole file, and both
were in the DSA indexer:

```c
for (unsigned st = blockIdx.x; st < nslab; st += nblk)   /* d_index_score_mfma  */
const unsigned bid = blockIdx.x, tid = threadIdx.x;      /* d_index_select_coop */
```

Every other kernel in `runtime/amd` takes `(slice, nblk)`. `op_kda.h:30` states the convention and
`test_kernels.hip`'s own header states the consequence: mapping `(blockIdx, gridDim)` onto
`(slice, nblk)` "is the only difference between a test launch and a packet."

**Under the persistent interpreter the workgroup that runs a stream entry is not the entry's logical
index.** `interp.hip` calls `plow_exec(in, e.slice, tens, &sm)` — the index the compiler assigned
(`crates/packet/src/devbuild.rs:1010`: *"`slice` is the op-local index of this workgroup, NOT the CU
id: the op's kernel splits its work into `blocks` shares and this is which share"*). And the **DECODE
PHASE DEFAULTS TO THE GLOBAL-QUEUE SCHEDULER** (`sched_decode = Sched::GlobalQueue`,
`crates/plowrt/src/exec/amd.rs:1656`), where entries are claimed off ONE shared atomic cursor by
whichever workgroup reaches it first. `blockIdx.x` is then unrelated to the slice.

Consequences, both silent:

* **INDEX_SCORE** (emitted all-CU, `nblk = 256`) — any workgroup that claimed two of the 256 entries
  ran the same `blockIdx.x` twice, so another index never ran and its slab of `Score[0, kv_len)` was
  **never written**. At a short context `nslab = ceil(533/256) = 3`, so the whole score array depends
  on workgroups 0, 1 and 2 specifically claiming an entry.
* **INDEX_SELECT** (emitted on 32 CUs, `nblk = 32`) — `bid` indexes the histogram pass, the histogram
  clear and the final emit loop. With `len = 533` only `bid` 0 and 1 own any rows at all
  (`t = bid*512 + tid`), so unless workgroups 0/1 happened to claim a select entry the emit loop
  wrote **nothing**, `idx[]` kept its bind-time contents, and every gathered row became latent
  **row 0**. Attention then returned the same vector for every head, on every layer, on every step.

That is the recorded symptom exactly: first token right (it comes from the DENSE prefill), every
later token wrong, identically at 532 / 2,032 / 8,017 / 118,792 prompt tokens, and independent of the
selection.

**`dsa_gather_bench` reported EXACT at 8k/32k/128k/256k and was not wrong** — a standalone launch has
`blockIdx.x == slice` by construction. This is the same blind-spot SHAPE as §3's `kv_len == ctx`
gate: the harness satisfied the broken invariant by construction.

The grid barrier survived because it counts arrivals rather than identities: `nwg = 32` arrivals still
happen, from 32 arbitrary workgroups. So the failure is silent corruption, not a hang. (`PLOW_GQ_BATCH`
must stay 1 for that to hold — a workgroup claiming two select entries would block in the first one's
barrier waiting for an arrival only it could make. It is 1, and `interp.hip:125` already required it
for an unrelated reason.)

**The STATIC scheduler was fine**, which is why this survived: both ops are emitted on CU sets that
start at 0 (`0..255` and `0..31`), so `slice == cu == blockIdx.x` there. `PLOW_STATIC_DECODE=1` is a
zero-rebuild confirmation of the diagnosis on any tree that predates the fix.

### 10.2 The fix

`slice` is now an operand of both kernels and `blockIdx.x` appears nowhere in `runtime/amd/*.h`.
`interp.hip` passes `slice`; the `test_kernels.hip` wrappers pass `blockIdx.x`, which is what a
standalone launch means, so `dsa_gather_bench` keeps measuring what it measured.

**Register delta: ZERO** — decode 248 VGPR / 0 AGPR / occ 2 / spill 0, prefill 256 / occ 2 / spill 2,
flash 512 / occ 1 / spill 228, byte-identical to the same tree without the change on all seven
audited objects. **Object-size delta: `interp_decode_gq.elf` 310,480 -> 310,416 B (-64 B, -0.02%);
`interp_decode.elf` +8 B; every other object byte-identical.** Nothing grows inside the persistent
megakernel, so §6g-GF8-REGRESSION's failure mode does not apply.

### 10.3 THE `kidx` DEFECT IS REAL, IT IS STILL OPEN, AND IT BOUNDS WHAT "DSA IS CORRECT" MEANS

§8's candidate verifies, and now from the SHIPPED PACKET rather than from the emitter. Scanning
`build.json` for the `l4-dsa-tp4` bundle: the four PREFILL buckets carry `FlashMlaPrefill` and
**no `IndexScore` / `IndexSelect`**; only the decode program has them. So `kv.{l}.kidx` is written at
exactly one site (`mla.rs:1927`, one row per decode step) and rows `0..P-1` after a P-token prefill
are whatever the allocator handed out. It is not even zeroed: `exec/amd.rs:1962` skips the memset for
every `kv.*` tensor, justified by *"attention reads only [0, kvlen), every row of which is written
before it is read"* — an invariant `ckv`/`krot` satisfy and `kidx` breaks.

**What that does and does not cost:**

* `kv_len <= index_topk` (2048): **nothing**. The selector emits every live row regardless of score,
  so the step is arithmetically dense. This is the whole regime the `l4` oracle tests.
* `kv_len > index_topk`: **the selection is drawn from scores computed against uninitialised keys.**
  The chosen rows are in range (so the output is coherent, not degenerate) but they are not the rows
  the indexer would have chosen. DSA is only a WIN above 2048, so this is exactly the regime that
  matters, and it means **no DSA quality claim above 2048 tokens is admissible yet.**

**NOW MEASURED, not inferred (§10.6):** at 7,054 prompt tokens the DSA arm recovers a needle at depth
0.0 and misses it at depths 0.5 and 0.9, returning the **byte-identical** wrong answer at both even
though the needle moved — the "oldest 2048 tokens" collapse, on hardware. Dense recovers it at all
three depths.

**What a fix needs (scoped, not built).** Per full-indexer layer, per prefill chunk of T rows:
`kidx_raw[T][128] = xn @ wk^T`, then `LayerNorm` (already T-row capable, `i[0] = t`), then the
interleaved RoPE into `kv.{l}.kidx` — and that last one needs NO runtime work, because
`kv_write_row_field` already rebases a `HeadNormRope` whose destination is `kv.*` (`i[3] = c0`).
The blocker is the GEMM: `wk` is block-fp8 `[128, 6144]` with a `[1][48]` `weight_scale_inv` grid, and
**there is no [128,128]-block-fp8 GEMM at M > 1 in the AMD op set** — `GemmFp8` (33) is w8a8 with
per-tensor/per-token scales, and the only block-fp8 prefill arm in the tree is the grouped-MoE pair
(ops 85/86, `d_moe_group_pf_t<FP8=true>`, `op_moe.h:975`, `KB = (K+127)>>7`). Three routes:

1. **Bind-time dequantisation of `wk` to bf16** and an ordinary `pick_tile` GEMM at prefill. 21 full
   layers x 128 x 6144 x 2 B = **33 MB** of extra device memory, no new kernel, no new opcode, and no
   register risk on a prefill object already at 256 VGPR / occ 2 / spill 2. Decode keeps using the
   fp8 tensor through `GemvFp8Blk`. **Recommended.**
2. A new `GemmFp8Blk` opcode reusing `d_gemm_t`'s tiling with `op_moe.h`'s block-scale walk. Correct
   in principle; it adds an instantiation to the object that is AT the cliff, so it must be
   register-checked, not assumed.
3. Reuse the grouped 1-expert path `bind_dense_ffn_tables` already sets up. No new kernel, but the
   plumbing is a routing table and a partial layout for what is a plain GEMM.

Arithmetic cost of the producer itself is negligible — 2*T*6144*128 FLOP per full layer against
projections two orders of magnitude larger — so TTFT should not move.

### 10.5 MEASURED, 2026-07-29, TP4, gfx950

One tree, one `plowrt` (`--features hsa`), one packet per arm, **two object directories built from the
same sources with and without the change** — so the only variable is the kernel edit.

| gate | before (`blockIdx.x`) | after (`slice`) |
|---|---|---|
| `l4` oracle, 3 prompts | **FAIL 3/3**, divergence at token 2 | **2 of 3 bit-identical over 32 tokens** |
| `l4` oracle, 12 prompts | — | **8 of 12 bit-identical**; 3 of the 4 divergences start after ~25 of 32 tokens |
| **full 78 layers, 5 prompts** | (§8: `'CR\nComments\nComments…'`) | **PASS — all 5 bit-identical AND coherent** |

The "before" column reproduces §9 **byte for byte** — same dense text, same DSA text — so the control
is exact rather than merely similar.

Full-78, DSA gate armed, greedy, `max_tokens=24`, identical to the dense arm on every prompt:

```
'Paris.'   '**Jupiter**.'   'The chemical symbol for gold is **Au**.'
'Water boils at **100 degrees Celsius** (at standard sea-level atmospheric pressure).'
"To calculate \(17 \times 23\), we can use the **distributive property of multiplication over addition**…"
```

**THE RESIDUAL `l4` DIVERGENCES ARE FLOATING-POINT REORDERING, NOT SELECTION, AND THE ORACLE CANNOT BE
MADE BIT-EXACT WITHOUT ONE MORE CHANGE.** Two controls establish this:

* **Reproducibility.** Three back-to-back runs: dense 1v2 and 1v3 IDENTICAL, DSA 1v2 and 1v3
  IDENTICAL. The residual is deterministic, so it is not a race in the selector.
* **`nsplit` is not the cause.** The DSA arm calls `glm_nsplit(itk=2048, …)` and the dense arm
  `glm_nsplit(ctx=135168, …)`, which at TP4/`nh_l=16` is **16 splits versus 64** over the same KV
  range — a different online-softmax merge tree and therefore a legitimate deterministic fp
  difference that has nothing to do with the selector. Pinning both arms to `PLOW_GLM_NS=16` and
  re-emitting produced a **byte-identical outcome**, so that is not it either.

What is left is the ORDER WITHIN a split: `d_index_select_coop` emits with
`slot = atomicAdd(&Cg[2],1)`, so `idx[]` holds the exact SET in arbitrary order, and the gather sums
those rows in that order while dense walks the cache ascending. Algebraically identical, not
bit-identical. **A wrong SET cannot produce 8 streams of 32 tokens that are bit-identical to dense**
(that is 256 consecutive decode steps x 3 full-indexer layers of selections), and the pre-fix arm
diverged at token 2 on every prompt.

The `l4` truncation is also a HARSHER gate than the real model, which is why the full-78 pair agrees
where the 4-layer one does not: a 4-layer truncation's logits are nearly tied, so a 1-ulp
perturbation flips the argmax constantly. At full depth the same perturbation flips nothing on any
of five prompts.

**To make the oracle exact, emit `idx[]` ASCENDING** — partition the emit pass by contiguous ranges
instead of a strided one, exclusive-scan the per-workgroup counts (one extra grid barrier and a
32-element scan on a kernel that already costs 53-79 us), and each workgroup writes its own positions
in order. Then at `top_k >= kv_len` the gather walks the cache exactly as dense does and the two arms
are bit-identical by construction. It is also independently worth having: a sorted gather list reads
the latent cache ascending. NOT done here — it is a change to the hot selector and wants its own
register and timing validation.

### 10.6 ABOVE `index_topk` DSA IS NO LONGER DEGENERATE BUT IS NOT SIGNED OFF — AND THAT IS §10.3

Needle-in-a-haystack, **7,054 prompt tokens** (3.4x `index_topk`), greedy, `max_tokens=24`, same
objects, same tree, needle walked through three depths:

| needle depth | dense (control) | DSA |
|---|---|---|
| 0.0 | `'CRIMSON-FALCON-77'` | `'CRIMSON-FALCON-77. The library was quiet that afternoon…'` |
| 0.5 | `'CRIMSON-FALCON-77'` | **`'CRASH! The library was quiet that afternoon and the voyages…'`** |
| 0.9 | `'CRIMSON-FALCON-77'` | **`'CRASH! The library was quiet that afternoon and the voyages…'`** — byte-identical to depth 0.5 |

**THIS IS §10.3, MEASURED.** DSA retrieves a needle at the FRONT of the context and misses it
everywhere else — and its two miss answers are byte-identical even though the needle moved, i.e. the
selection never reached it in either case. That is the exact signature predicted for a `kv.{l}.kidx`
that reads as zero: every prefill position scores `sum_h w[h]*ReLU(q.0) = 0`, the tie-break is lowest
index, and the selection collapses to **the OLDEST 2048 tokens**. Depth 0.0 "passes" for entirely the
wrong reason.

It is still a different world from §8's `'CR\nComments\nComments…'` — the path is structurally sound
now and the failure is a bad SELECTION rather than an out-of-range gather. But it is emphatically
**not** evidence that DSA is correct above 2048.

**No DSA TPOT number is quoted here, and none should be.** Oracle agreement exists only below
`index_topk`, and below `index_topk` DSA is a measured LOSS (0.98x at 8k, §6). The regime where the
2.20x / 7.40x / 14.1x of §6 would apply is exactly the regime §10.3 leaves unvalidated. Per the MoE
confound a subtly-wrong DSA arm OVER-REPORTS, so a timing taken now would be worse than no timing.

### 10.7 A PREFILL GATHER IS A NEW KERNEL, NOT A WIRING JOB

Confirmed from the same `build.json`: `FlashGatherPrefill` appears in NEITHER arm, at any bucket.
**DSA is decode-only in this tree and could never have moved TTFT.** What a prefill gather would need,
scoped:

* `IndexScore` at T queries (`Score[T][ctx]`, `Qidx[T][HI][DI]`). The cheap half — the MFMA form
  already streams the key tile through LDS and re-runs it per query, so the tile is amortised across
  the queries in a chunk and it should get *faster* per token (mla.rs:2972 says the same).
* `IndexSelect` at T independent top-k selections. The current kernel is ONE grid-wide cooperative
  radix per query; T of those in series is T grid barriers and is unusable at T = 8192. It needs a
  per-query, workgroup-local selection — a different algorithm, not a new call site.
* **The score array cannot be materialised.** `Score[T][ctx]` at T=8192, ctx=131072, f32 is **4 TB**.
  The selection has to be FUSED into the score kernel (running per-query top-k over key tiles) or the
  chunk has to be far narrower than the 8192 the ladder uses.
* Causality: each query's selection must be restricted to `pos <= qpos`. The decode selector has
  never needed a mask.
