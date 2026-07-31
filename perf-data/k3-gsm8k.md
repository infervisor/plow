# K3 GSM8K — the first accuracy number for plow, and how it was obtained

**Measured 2026-07-30.** Kimi-K3 93 layers (69 KDA + 24 MLA), TP8 over 8× gfx950 (MI355X),
fp8 KV + mxfp4 experts, served by `plowrt serve` from this tree, scored by
`scripts/bench_gsm8k.sh`.

```
GSM8K  8-shot  greedy(temp=0.0)  n=200  errors=0
  exact_match = 155/200 = 0.7750
  latency/question: median 4.23s  mean 4.46s  total 892s
```

A 4-shot smoke run on the same server scored 4/4 and the coherence gate passed on both.

## Why this number did not exist before

Every gate in the tree before this one is a **token-identity** gate — `k3_tp_equivalence.sh`
(tp=1 vs tp=8 on one asset), the Paris continuation in `kimi-k3-README.md` §4, the coherence check
inside `bench_plowrt_serve.sh`. Each proves the runtime is *self-consistent*. None of them proves
the model is *right*: a blob that is wrong in the same way on every rank passes all three. A
throughput number without an accuracy number is not publishable against vLLM, and this closes it.

It also exercises the whole serving stack rather than the decode kernel — chat template, K3's
`<|close|>` channel stop, the tokenizer, prefill over ~1k-token 8-shot prompts, on-device
sampling, and SSE — in a way `amd-bench` cannot.

## Method, and two choices that are not simplifications

* **Greedy, `temperature=0`.** plow's gfx950 backend samples argmax ON DEVICE and the host never
  sees the logit row, so `top_p`/`top_k`/penalties are ignored on this backend entirely. Reporting
  a sampled number would be reporting something the backend cannot produce.
* **Exact match on the final number** — last number in the completion, commas and a trailing
  period stripped, against the token after `####`. This is lm-eval's `gsm8k` convention and is
  deliberately NOT a "contains" match, which scores a model that emits the right digits inside a
  wrong derivation.

## Known limitations of this run, stated rather than buried

1. **n=200 of the 1319-question test split.** At 4.23 s/question the full split is ~93 min of
   leased GPU. The sampling error on 200 at p≈0.78 is ±2.9pp (1σ), so quote this as **77.5 ± 3**,
   and do not compare it to a published figure closer than that without running the full split.
2. **One completion produced no parseable number** (`got=None` at question 180) and was scored
   wrong. That is the correct conservative call, but a model that ran out of `MAXTOK=320`
   mid-derivation is penalised the same as one that answered incorrectly.
3. **No vLLM side-by-side.** The number is only comparable against a same-prompt, same-shots,
   same-parser run on the reference stack. `bench_gsm8k.sh` takes any OpenAI endpoint, so that run
   is one command — it just has not been done.
4. **Batch 1.** K3 decode is structurally single-sequence (KDA recurrent state has no batch axis),
   so this is 200 sequential requests. It measures accuracy correctly and says nothing about
   accuracy under concurrency.

## Reproduce

```bash
perf-data/harness/gpulease -n 8 gsm8k sg render -c \
  "N=200 SHOTS=8 MAXTOK=320 PLOWRT_BIN=<hsa-built plowrt> \
   scripts/bench_gsm8k.sh <assets> 8412 auto 1800"
```

`<assets>` needs `model.pkt`, `hsaco/`, `checkpoint/`, `tokenizer.json`. `MODEL=auto` resolves the
slug from `/v1/models` — the served id comes from the blob's network name, and guessing it wrong
fails the coherence gate in a way indistinguishable from a bad model (observed on the first try:
`no model registered for 'auto'`).

The `sg render -c` above is load-bearing and not decoration: `/dev/kfd` is `root:render 0660`, and
a shell whose process credentials lack the `render` group gets `hsa_init failed: 4104` and
**silently falls back to the CPU reference backend**. Observed again 2026-07-31 — the coherence
gate caught it, which is the only reason it did not become a bogus accuracy number.

# 2. BATCHED DECODE SCORES HIGHER THAN THE SHIPPED B=1 PATH (measured 2026-07-31)

Limitation 4 above ("Batch 1 ... says nothing about accuracy under concurrency") is now closed, and
the answer was not the expected one. All three runs are `N=200 SHOTS=8 MAXTOK=320 TEMP=0`, same
question set, same order, same checkpoint (`k3_farm`), branch `k3-batched-decode`.

| run | decode | MoE kernel | conc | exact_match | errors | median s/q |
|---|---|---|--:|--:|--:|--:|
| §1 baseline, 07-30 | B=1 | per-slot | 1 | 155/200 = **0.7750** | 0 | 4.23 |
| control, 07-31 | B=1 | per-slot | 1 | 162/200 = **0.8100** | 0 | 4.37 |
| **isolating, 07-31** | **B=4** | **grouped** | **1** | **168/200 = 0.8400** | 0 | 9.17 |
| batched, 07-31 | B=4 | grouped | 4 | 177/196 = **0.9031** | **4** | 12.94 |

## 2.1 THE ISOLATING RUN REFUTES THE KERNEL HYPOTHESIS

The first two runs alone invited the reading "the grouped expert kernel is worth +9 points, so the
SHIPPED per-slot decode path is losing accuracy." **That reading is wrong, and the control is what
shows it.** Same B=4 packet, same grouped kernel, sequential requests:

| comparison | isolates | Δ | z (unpaired) | verdict |
|---|---|--:|--:|---|
| B=1 c1 → B=4 c1 | **the MoE kernel** | +3.0pp | **0.79** | **not significant** |
| B=4 c1 → B=4 c4 | concurrency only | +6.3pp | 1.89 | not significant (p~0.06) |
| B=4 c1 → B=4 c4, errors scored WRONG | concurrency only | +4.5pp | 1.31 | not significant |
| B=1 c1 → B=4 c4 | both at once | +9.3pp | 2.67 | significant, but confounded |

The only comparison that reached significance is the one that moves BOTH variables. Isolate either
and it vanishes. **There is no established accuracy difference between the grouped and per-slot MoE
paths**, and the earlier +9.3pp framing does not survive its own control.

Note also that the c4 arm's 0.9031 is over n=196: the 4 excluded requests are the §9 divergence, and
excluding them can only inflate the score. Scored as wrong it is 0.885, and the concurrency step
drops to z=1.31.

## 2.2 The spread across nominally identical runs is the real finding

Four runs, one question set, greedy at temperature 0, span **77.5 - 90.3%**. Two of them are B=1 at
the same documented settings and differ by 3.5pp.

**Greedy decoding on a fixed question set should be REPRODUCIBLE.** Two candidate explanations, and
this file does not yet distinguish them:

1. The 07-30 baseline ran on a different tree state and different assets. Plausible and boring.
2. There is genuine run-to-run nondeterminism. This is NOT speculation: §9 of
   `perf-data/k3-batched-decode-design.md` records a MEASURED cross-rank divergence in the
   `d_xargmax_fin_mega` fold — a race that changes a sampled token. A race that changes tokens
   changes accuracy, and it would make greedy runs non-reproducible by construction.

The distinguishing experiment is cheap and has not been run: **the same assets, twice, back to
back.** Identical scores implicate (1); differing scores implicate (2) and make the §9 race a
first-order correctness item rather than a rare-hard-fail item.

## 2.3 RESOLVED — every number above was depressed by a serving bug. K3 scores 98.0%.

**The spread §2.2 could not explain had one cause, and it was not sampling noise.** `begin_slot` —
the function that clears a slot's carried KDA recurrence when it is handed to a new request — was
called only on the single-GPU path. K3 serves at **TP8**, so on the shipped configuration it never
ran at all, and every request after the first on a slot began from its predecessor's accumulated
recurrent state across **69 of K3's 93 layers**.

Fixed (per-slot clear, applied on every rank). Same assets, same harness, same settings:

| run | decode | carried state | exact_match | Δ vs its control |
|---|---|---|--:|--:|
| control | B=1, conc 1 | **leaked between requests** | 162/200 = 0.8100 | — |
| **with the fix** | B=1, conc 1 | **cleared per slot** | **196/200 = 0.9800** | **+17.0pp** |
| control | B=4, conc 4 | leaked | 177/196 = 0.9031 | — |
| **with the fix** | B=4, conc 4 | **cleared per slot** | **193/199 = 0.9698** | **+6.7pp** |

**B=1: z = 5.77, p < 1e-8.** Not a sampling artefact and not a marginal effect.

The B=4 arm is the one that validates the PER-SLOT striding under real concurrency — a whole-tensor
clear would have wiped live slots and shown up as a collapse, not a 6.7pp gain. Batched decode now
matches single-sequence within sampling error (0.9698 vs 0.9800, z = 0.63), which is the result the
batched work was supposed to produce all along.

It also explains §2.2 retrospectively. Contamination depends on what the PREVIOUS request left in
the state, so the score depends on request ordering and on how far each run happened to drift —
which is exactly the "greedy runs that should be reproducible but are not" signature. The 77.5 /
81.0 / 84.0 / 90.3 spread was four different draws from a broken serving path.

**Quote plow's K3 GSM8K as 98.0% (196/200, 8-shot, greedy, n=200).** Every earlier number in this
file is superseded and should be read only as the record of the bug.

## 2.4 What this says about the earlier analysis

Worth stating plainly, because the earlier sections argued the opposite:

* The "grouped MoE kernel is more accurate" hypothesis was already refuted by its own isolating
  control (§2.1, z=0.79). That refutation stands — the variation was never about the MoE kernel.
* The residual "unexplained spread" flagged in §2.2 named two candidates: different assets, or
  genuine nondeterminism from the §9 cross-rank race. **Both were wrong.** The cause was a
  deterministic state leak, and neither candidate would have found it — it was found by reading the
  slot lifecycle for a throughput review, not by chasing the accuracy number.
* The §9 cross-rank divergence is a SEPARATE, still-open defect. It is rare (1 event in ~1e4-1e5
  steps) and it hard-fails a request rather than silently degrading one; it is not this.

## 2.4 The 4 errors are one fault, not four

All four failures are a single cross-rank divergence in one decode step
(`perf-data/k3-batched-decode-design.md` §9). At B>1 every in-flight request shares the step, so
one bad step fails the whole batch — which is why they arrive as a burst of exactly B. Both CONC=1
runs had **zero** errors and zero divergences over 200 sequential requests each, which is
consistent with the race needing concurrent slot occupancy to show up, but 2 runs is not evidence
of a rate.

## 2.2 The 4 errors are one fault, not four

All four failures are a single cross-rank divergence in one decode step
(`perf-data/k3-batched-decode-design.md` §9). At B>1 every in-flight request shares the step, so
one bad step fails the whole batch — which is why they arrive as a burst of exactly B. The B=1
control had **zero** errors and zero divergences over 200 sequential requests.
