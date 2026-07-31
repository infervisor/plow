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
