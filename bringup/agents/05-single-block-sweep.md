# Agent — Stage 5: Single-Block Sweep

You are executing **Stage 5** of the model-bringup playbook. Your job: take one
transformer block of the target model, **prove it is numerically correct**, then
**tune its isolated latency to the per-block target**, and decide whether the
model is ready for Stage 6 (runtime optimization). Read
`docs/bringup/05-single-block-sweep.md` first — it has the full methodology,
commands, and pitfalls. This prompt is the executable checklist.

## Preconditions (from Stages 1–4)

* The model's config + geometry are understood (hidden, head counts, head_dim,
  intermediate size, expert layout, which family: Gemma-dense / Gemma-MoE / GLM
  MLA+MoE / Kimi-K3 / DeepSeek / Nemotron).
* `plowc --emit devblob` produces a full-model `model.pkt` without aborting.
* You know your target arch/GPU (`--arch`, `--gpu`, `--n-cu`) and have a GPU to
  run on. Use `nix develop` for all build/cargo commands.

If any is missing, **stop and report** — do not start Stage 5 on an unvalidated
emit.

## Identify the family first — it determines the whole path

Two block shapes exist, and they use different runners:

* **Gemma / dense-attn path:** the `--block` asset carries `embed_tokens` +
  lm_head → it is model-shaped and driven directly by `plowrt amd-bench` /
  `examples/block_run`.
* **MLA family (GLM / Kimi / DeepSeek / Nemotron):** the `--block` asset emits
  only `act.x`-in / `act.x`-out (no embed, no lm_head, no `act.logits`). It is
  driven by `plowrt amd-block` (A/B) or a **C oracle harness** in
  `runtime/tests/`. `amd-bench` cannot drive it; for latency use a
  truncated-model sweep (`PLOW_GLM_LAYERS=N` / `K3_NLAYERS=N`).

Confirm which you have with `plowrt amd-block --list-tensors` or
`plowrt disasm <model.pkt>` before proceeding.

## Procedure

### 1. Emit and inspect one block

```bash
plowc --emit devblob --block 0 --max-ctx <ctx> --n-cu <N> --gpu <gpu> \
  --hf-dir <ckpt-dir> --out <asset>/
plowrt disasm <asset>/model.pkt --program 1      # confirm the expected ops are present
```

Check the sibling `block.json` records the geometry you expect. `--block l..r`
extracts a layer range if you need more than one layer (e.g. to include the
first layer of a second mixer type).

### 2. CPU sim — does it plan?

```bash
./scripts/block_sim.sh <ckpt-dir-or-descriptor> prefill:1:128
```

Packets fire, makespan finite, op-family counts sane. A block that will not sim
will not run — fix planning failures here, cheaply, before touching a GPU.

### 3. Correctness (THE GATE — do this before any timing)

Pick the mechanism for the family:

* **C oracle harness** (authoritative, vs HF/host reference). E.g. GLM-5.2:
  ```bash
  nix develop --command bash -c './scripts/build_glm52_block.sh /tmp/glm52b'
  sg render -c 'cd /tmp/glm52b && /usr/bin/env -i PATH=/usr/bin:/bin HOME=$HOME \
      LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=<n> \
      ./glm52_test interp_decode.elf glm52_fixture.bin'
  ```
  Use the matching `runtime/tests/<family>_block_*_test.c` + `<family>_oracle.py`.
  Read the **per-substep** diffs (attn_out, x_mid, xn2, router top-8 set + gates,
  x_next), not just the block boundary — a failing substep localizes the wrong op.

* **`amd-block` A/B** (precision deltas, weight-free): compile the block at two
  precisions, dump both, diff.
  ```bash
  plowrt amd-block --blob <asset>/model.pkt --hsaco <hsaco> --checkpoint <ckpt> \
    --inspect act.x,act.hn --dump /tmp/out.bin
  ```
  `--inspect` catches silent zeros (unwritten output). `block_run … check` is a
  smoke test only (shape/finiteness) — NOT parity.

**Tolerances:** bf16+MoE ~1e-2, block-fp8 ~3e-2. A bit-exact match is expected
only for a same-precision same-weight A/B.

**Do not proceed to timing until correctness passes.** A fast wrong block is
worthless.

### 4. Latency — bench and sweep

* **Gemma path:**
  ```bash
  ./target/release/examples/block_run <asset> bench \
    --batch 1,4 --ctx 128,1024,4096 --iters 100 --warmup 20 --prefill-iters 10
  ```
* **MLA family:** truncated-model sweep — `./scripts/k3_block_sweep.sh` (Kimi),
  `./scripts/glm52_block_sweep_gfx942.sh` (GLM). Apply emit knobs as
  `KNOB=val ./scripts/k3_block_sweep.sh`.

Every timed run must be uncontended (`gpulease` rc=0). Discard and re-run any
rc=76.

### 5. Anchor and attribute

* Anchor the isolated per-block number against a reference-framework floor:
  `per_block_us ≈ (TPOT_ms − fixed_overhead_ms) × 1000 / num_layers`. Run the
  reference side with `scripts/block_layer_bench.py` / `block_baseline.py`, join
  with `scripts/block_compare.py --phase {decode,prefill}`. Check
  `block_us × layers ≈ decode body`.
* Attribute per-op with `scripts/block_op_bench.py --phases decode,prefill` (the
  reference target by kernel) and a `-DPLOW_NV_TRACE=1` cubin +
  `trace_summary` in `block_run` (plow's own breakdown).

## Interpreting results

* **Correct + at target →** gate passes. Go to Stage 6 with the per-op profile;
  name the largest single op (commonly `moe_experts`, then `qkv_proj`) as the
  first optimization prize.
* **Correct + slow →** report the gap **as a %-of-HBM-peak (decode) or TFLOP/s
  (prefill) per projection**, not a single latency. A roughly uniform gap across
  projections points at a shared occupancy ceiling (attack blocks/SM globally),
  not one bad kernel. Hand this to Stage 6.
* **Incorrect →** stop. Localize with the per-substep diff, fix the emitter/op,
  re-run Step 3. Do not tune a wrong block.

## Pitfalls to actively guard against

* Block ms ≠ model ms — truncated-model scores are **rankings**, not TTFT.
* Isolated whole-block us **over-states** the norm/launch floor (~180 us of
  launch overhead plow fuses away). Trust the per-op GEMV/attn rows.
* An **eager** reference baseline flatters plow ~2.5× — the reference must
  CUDA-graph the decode step.
* Few-instances-per-block ops (router/combine) sink into timer noise on a block —
  defer them to the full network.
* A knob measured on an isolated kernel can **invert** in the megakernel — prefer
  a truncated-model blob for ranking runtime knobs.
* Reference geometry may be an approximation (MoE expert count / top-k) — fix the
  descriptor before comparing MoE.
* Cross-GPU numbers are not comparable; re-anchor per card.
* The tune system may be a no-op on your GPU/precision (aliased GEMM, no decode
  GEMV kernel) — confirm it selects something before calling the block "tuned".
* Driver/toolchain mismatch (`CUDA_ERROR_INVALID_IMAGE`, "driver too old") fails
  runs for reasons unrelated to the block.

## When to stop and ask

* The full-model emit aborts, or the block asset lacks tensors you expect
  (`--list-tensors` / `disasm` disagree with the config) → a Stage 1–4 defect.
* No C oracle harness / oracle fixture exists for this family in
  `runtime/tests/` → correctness cannot be established in-tree; report and ask
  before hand-rolling one.
* Correctness fails and the per-substep diff does not localize a single op.
* Every timed run is contended and you cannot get an uncontended card.
* The measured gap is not uniform and not attributable to one op — surface the
  per-op table and let Stage 6 triage.

## Report back

* **Family + block shape** (model-shaped vs act.x-only) and how you drove it.
* **Correctness:** pass/fail, tolerance used, per-substep results, any silent
  zeros found.
* **Latency:** decode us / prefill ms sweep, the anchored per-block floor, and
  whether `block_us × layers` reconciles.
* **Per-op profile** and the single biggest op.
* **Gate decision:** ready for Stage 6, or blocked (with the specific blocker).
* **Real-vs-ideal caveats** that affect trust in the numbers (contention,
  geometry approximation, cross-GPU anchor, toolchain).
