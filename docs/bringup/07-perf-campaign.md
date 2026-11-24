# Stage 7 — Performance campaign

Measure the gated recipe against its vLLM peer, behind the recipe's accuracy gate, and record the
result. This stage measures; it does not tune. A run that suggests a lever is open work for
[Stage 5](05-single-block-sweep.md), [Stage 6](06-runtime-opt.md) or
[Stage 8](08-rung-campaign.md).

**Precondition:** Stage 6 gated out; the recipe is a `recipes/infervisor/<model>/<label>.toml`
built by `campaign.py build`; the [`target.md`](target.md) block is filled in and goes into the
campaign header. Only block-level winners reach this stage ([agent-tools.md](agent-tools.md)
"HARD RULE"); the report opens with the block-vs-roofline table.

**Gate out (bringup complete):** `llm_fp32_ref` passes on the served packet at every recipe
concurrency, both serving arms ran on the same client and traffic, and `campaign.py report`
rendered the strict comparison into `docs/bringup/results/<campaign>/`.

Cross-cutting rules: order of operations [agent-tools.md](agent-tools.md) "Campaign playbook";
queue and leases "Queue"; judging and the vLLM peer "Working rules"; repeats and noise "A/B rule";
inherited env "Measurement hazards".

## Plan

Write down before any GPU time: model and precision, the recipe path, input/output lengths, the
concurrency ladder, traffic profiles (single-turn, `--agentic`, `--prod` with `PROD_RATES`), SLOs,
and the vLLM peer configuration (same checkpoint, same precision, prefix caching on). Every SLO is
a number for `$GPU`; one taken from another part is a hypothesis.

## Steps

All GPU steps run through `python3 scripts/bench/gpuq.py submit <label> <ngpu> <command...>`.

1. **Preflight (CPU).** `scripts/bench/plowbench-doctor.sh <assets> <objdir>`.
2. **Accuracy gate.** On the exact packet you will serve:

   ```bash
   python3 scripts/campaign/campaign.py gate <recipe> --assets <assets> --out <gate> --only llm_fp32_ref --dry-run
   python3 scripts/bench/gpuq.py submit gate <ngpu> bash <gate>/run.sh
   python3 scripts/campaign/campaign.py gate <recipe> --assets <assets> --out <gate> --only llm_fp32_ref --score-only
   ```

   A screen runs one c16 gate. Promotion runs the recipe's `[gates.llm_fp32_ref].concurrency`,
   which includes c1 and every served concurrency, each with a same-concurrency vLLM peer.
   Thresholds and relaxation: [agent-tools.md](agent-tools.md) "Gates". No serving cell is
   reported for a packet that has not passed this gate.
3. **Serving arms.** Same client, traffic seeds, concurrency, GPU count and cache policy on both:

   ```bash
   python3 scripts/bench/gpuq.py submit grid-plow <ngpu> env ASSETS=<assets> HF=<hf> KV_DTYPE=<dtype> \
       SERVE_ARGS="<recipe serve flags>" scripts/bench/llm_grid.sh plow <res>/plow [--agentic|--prod]
   python3 scripts/bench/gpuq.py submit grid-vllm <ngpu> env HF=<hf> VLLM_ARGS="<matched flags>" \
       scripts/bench/llm_grid.sh vllm <res>/vllm [--agentic|--prod]
   ```

   Reuse a captured vLLM peer whose recorded provenance (GPU, vLLM version and flags, client,
   traffic seeds) matches the plow arm; never recapture it for a screen.
4. **Audit.** `python3 scripts/bench/waterfall.py <res>/plow <res>/vllm` — spread per cell, reference
   prefix-cache hits, wall split, decode ms/step. Resolve a failed audit before reporting.
5. **Report.**

   ```bash
   python3 scripts/campaign/campaign.py report --baseline <res>/vllm --infervisor <res>/plow \
       --gate <gate>/gates.json --out <dir> [--cells <tags>]
   ```

   This is the only accepted final comparison format ([agent-tools.md](agent-tools.md)
   "Final performance report (strict)"). Never hand-write the table.

## Results placement

* Raw logs, JSON, captures, traces and per-tool result trees stay in `$RESULTS`, outside the repo.
* The campaign record is `docs/bringup/results/<campaign>/comparison.md` (rendered report plus
  qualified wins with evidence links) and its one `comparison.csv`; index new campaigns in
  [results/README.md](results/README.md).
* `perf-data/campaign/` holds only reference CSVs for `campaign.py compare`.

## Pitfalls

* `$MAXCTX` must hold `input + output` of every cell; otherwise the request is refused and the
  cell shows collapsed tokens/request. That is a compile-size error: re-emit with a larger
  `--max-ctx`, do not re-run.
* Device init failure falls back to the CPU reference with correct text. Confirm
  `backend ready — GPU accelerated` in every plow server log.
* Compare total generated tokens across arms; an `ignore_eos` or output-cap mismatch distorts
  throughput. Shed requests count as successful with few tokens — raise `--slo-ms` for throughput
  cells.
* A gate on a stored asset cannot catch an emitter regression; gate the packet emitted from the
  current checkout.
* Lossy modes (FP8 KV, FP8 activations) are compared only against a peer of the same precision.

## Stop and ask

* A target-block row cannot be filled, or `$MAXCTX` cannot cover the planned ladder.
* `llm_fp32_ref` fails, or TP ranks disagree — a correctness blocker; report no perf number.
* No vLLM peer can serve this model on `$GPU` — the campaign makes no comparative claim.
* The target needs a structural change (decode width, TP/DP, a different part) — a Stage 5/6
  decision.
