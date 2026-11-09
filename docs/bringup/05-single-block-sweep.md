# Stage 5 — Single-block sweep

Prove one block numerically correct, then tune its isolated latency against that block's roofline,
before any whole-model work.

- Rule: [agent-tools.md](agent-tools.md) "HARD RULE" — every kernel, fusion, route or knob candidate is
  screened here at the served shape, precision, flags, rung width and context; the report starts with
  the block-vs-roofline table. A candidate that hits a listed harness gap says so in its report.
- Precondition: Stages 1–4 green; `plowc --emit devblob` produces a full-model `model.pkt`; the
  [`target.md`](target.md) block is filled in. Every command uses `--gpu $GPU --arch $ISA --n-cu $NCU`.
- GPU access: [agent-tools.md](agent-tools.md) "Queue". Repeats and noise: [agent-tools.md](agent-tools.md) "A/B rule".
- Gate out (to [Stage 6](06-runtime-opt.md)): the block matches a host reference within tolerance
  **and** its isolated latency is at or near the per-block floor measured on `$GPU`.

Correctness is target-independent (host oracle). Latency is a statement about `$GPU` only. Do
correctness first.

Anchor both ways: `block_us × num_layers ≈ decode body`, and
`per_block_us ≈ (TPOT_ms − fixed_overhead_ms) × 1000 / num_layers`. Agreement means the isolated
measurement is trustworthy.

## Measurement tiers

A knob's tier is a property of the knob, not of the budget. This stage is tier 2.

| tier | harness | use |
|---|---|---|
| 1 prune | standalone probes: `runtime/tests/`, `runtime/bench/`, `runtime/nvidia/experiments/` | correctness-gate a rung, kill bad shapes; never trust its ranking |
| 2 sweep | `plowc --emit devblob --block` + `block_run bench` / `campaign.py block-bench`, or a truncated-model sweep | the wide sweep: real engine, real megakernel |
| 3 confirm | `examples/step_bench` — whole packet, real layer mix | 2–3 finalists only |
| 4 accept | serving client against `plowrt serve` ([Stage 7](07-perf-campaign.md)) | only tier that sees host-gap knobs |

```bash
nix develop --command cargo build --release -p plowrt $FEATURES --bin plowrt --example step_bench
PLOW_STEP_TIME=1 ./target/release/examples/step_bench <assets> [slots] [ctx] [steps]   # via gpuq
```

A probe re-runs one kernel on cache-resident inputs with no co-scheduled ops; a probe win can be a
loss in-model and vice versa. A probe may motivate or prune a variant; only an in-model gate decides.

Tier 2 cannot score:

- Prefill dispatch/classing knobs (per-object block size, segment classing, segment graphs) — the
  block harness drives the decode/single-segment path.
- Prefill bucket policy (`PLOW_PF_LADDER_APPEND`) — needs varied prompt lengths, i.e. a client.
- Host-gap behaviour (objective-derived multistep, device sampling, `--slo-ms`) — between kernels.
- `PLOW_MAX_CHUNK` — an analytic memory-sizing decision ([Stage 6](06-runtime-opt.md)); do not bench it.

Harness caveats: `block_run --batch` selects active slots, not kernel width — emit one block per
`PLOW_DECODE_BATCH` you score. Layer kinds are not interchangeable — emit one block per kind and rank
on the kind-weighted marginal cost `N_a·L_a + N_b·L_b` (fixed overhead and lm_head cancel).

## Select the existing harness

Search `runtime/bench/`, `runtime/tests/`, `scripts/`, `scripts/campaign/` and `perf-data/tools/`
first; name the selected harness in the report. Extend the closest one when its boundary is
incomplete; never add a model- or experiment-named runner to vary flags.

| need | harness |
|---|---|
| packet planning | `scripts/block_sim.sh` |
| block numerics | family C oracle in `runtime/tests/`, or `plowrt amd-block`; `block_run check` is smoke only |
| broad NVIDIA object/packet grid | `scripts/tune_decode_sweep.sh --block <l> --block-run <block_run>` |
| exact rung kernels (`sm*` and `gfx*`) | `scripts/gemma4_h100_kernel_tuner.py` (generic despite its name) |
| model-shaped block timing (NVIDIA) | `examples/block_run bench` / `prefill-bench`; A/B via `campaign.py nv-block-ab` |
| act.x-only block timing (AMD) | `campaign.py block-bench` / `block-ab` (section below) |
| MLA-family truncated-model timing | `scripts/k3_block_sweep.sh`, `scripts/glm52_block_sweep_gfx942.sh` |
| reference block and per-op floor | `scripts/block_baseline.py`, `scripts/block_layer_bench.py`, `scripts/block_op_bench.py`, then `scripts/block_compare.py` |
| block roofline | `scripts/campaign/block_roofline.py` (NVIDIA), `campaign.py block-roofline` (AMD) |
| finalist confirmation | `examples/step_bench` |

## The `--block` emit path

```bash
plowc --emit devblob --block <l> --arch $ISA --gpu $GPU --n-cu $NCU \
  --max-ctx $MAXCTX --num-gpus $NGPU --hf-dir <ckpt> --out <asset>     # --block l..r = [l, r)
python3 scripts/campaign/campaign.py build recipes/infervisor/<model>/<label>.toml \
  --block 4..6 --out <fresh dir>          # recipe-exact env, objects, routes; asset = <fresh dir>/assets
```

`devgen::block::parse_block` parses the range (`PLOW_BLOCK` is the env fallback);
`write_block_descriptor` writes a sibling `block.json` with the block geometry.

| family | emitter (`crates/devgen/src/`) | block program |
|---|---|---|
| Gemma-4 dense / MoE | dense-attn `--block` path (`lib.rs`) | carries `in.ids`/`embed_tokens` + lm_head: model-shaped |
| GLM (`glm_moe_dsa`) | `glm_emit_block` → `glm_build_block` / `glm_build_block_pf` (`mla.rs`) | `act.x` in / `act.x` out, no embed, no lm_head |
| Kimi-K3 | `kimi_emit_block` (`mla.rs`) | `act.x` in / out |
| Nemotron-3 | `nemotron_emit_block` (`mla.rs`) | `act.x` in / out; only the `--block` device path is wired |

Model-shaped blocks run under `examples/block_run` (NVIDIA). act.x-only blocks run under
`plowrt amd-block` / `campaign.py block-bench` or a family C oracle. Confirm which you have before a GPU:

```bash
plowrt disasm <asset>/model.pkt --program 1        # decode program; --range 0..40 --format json
plowrt amd-block --blob <asset>/model.pkt --hsaco <objects> --list-tensors
```

## Step 1 — CPU sim

```bash
./scripts/block_sim.sh crates/plowc/examples/transformer_block_gemma4_12b.json prefill:1:128
./scripts/block_sim.sh <hf-dir> decode:1:128
```

Compiles one block and runs `plowrt simulate --assets <out> --bucket <phase>:<batch>:<seq>`
(`--math golden` for reference numerics). Packets fire, makespan finite, op-family counts sane. A
block that will not sim will not run.

## Step 2 — correctness (the gate)

**C oracle harness** (authoritative, vs HF/host reference): `runtime/tests/<family>_block_*_test.c`
with its `<family>_oracle.py` fixture — GLM (`glm52_block_gfx950_test.c`, `glm52_real_block_gfx950_test.c`,
`glm_block_gfx950_test.c`), Kimi-K3 (`k3_*block_gfx950_test.c`), DeepSeek, generic MoE, block-fp8,
Gemma (`net_gemma_block_test.c`). The `_gfx950_` suffix names the first ISA, not a restriction.

```bash
nix develop --command bash -c './scripts/build_glm52_real.sh <out>'      # interpreter + host harness
nix develop --command python3 runtime/tests/glm52_real_oracle.py <out>/glm52_real_fixture.bin
nix develop --command python3 scripts/bench/gpuq.py submit glm52-oracle 1 <run script>
# run script: the clean-env harness line from the build script's header
```

Read the per-substep diffs (`attn_out`, `x_mid`, `xn2`, router top-k set + gates, `x_next`), not
only the block boundary. The GLM harness runs twice: BF16 routed experts, then block-fp8.

**`amd-block` A/B** (precision delta): same block, same weights, two precisions.

```bash
plowrt amd-block --blob <asset>/model.pkt --hsaco <objects> --checkpoint <ckpt> \
  --inspect act.x,act.hn --dump <scratch>/out_bf16.bin       # run through gpuq
```

`--inspect` catches silently unwritten outputs; `--dump` gives a bit-exact diff between two builds.

- Tolerances: BF16 + MoE ~1e-2; block-fp8 ~3e-2; bit-exact only for same-precision, same-weight A/B.
- A new AMD precision, split-K, attention or grouped-MoE arm: cover ragged M/K, empty experts and
  maximum top-k, batch 1 and the widest decode rung, context below/at/above tile and KV-ring
  boundaries, repeated deterministic runs. For TP compare TP1 with the target rank count and require
  rank token identity.
- `AMD_SERIALIZE_KERNEL=3` / `HIP_LAUNCH_BLOCKING=1` only localize a fault; never time with them.

Do not time a block that fails this step.

## Step 3 — latency

Decode µs per step and prefill ms per pass over a batch × context grid, device-timed. Dense blocks
are timing-insensitive to weight values. MoE blocks are not: price the expert union from a recorded
router table (`scripts/bench/router_table.py` on a `PLOW_NV_DUMP_ACT` dump, then
`block_roofline.py --router-table`) and confirm MoE levers with `step_bench` at the rung.

### NVIDIA model-shaped block

```bash
./target/release/examples/block_run <asset> bench \
  --batch 1,4 --ctx 128,1024,4096 --iters 100 --warmup 20 --prefill-iters 10
./target/release/examples/block_run <asset> prefill-bench \
  --query-rows 512,2048 --prior 0,4096,16384 [--requests 4 --riders 16 --rider-ctx 4096]
python3 scripts/campaign/campaign.py nv-block-ab --control <prod block> --treatment <cand block> \
  --args "--batch 1,8 --ctx 4096 --iters 200" --ceilings <ceilings.json> --out <scratch> --note "<lever>"
```

`block_run` writes `sweep.json` (packet, binary, device, routes). `nv-block-ab` runs
control/treatment/control/treatment in one queued lease and prints the HARD RULE table with a
per-point verdict.

### MLA family: truncated-model sweep

`PLOW_LAYERS=N` emits a few real layers of the real megakernel at the served TP geometry and
differences the fixed overhead out.

```bash
./scripts/k3_block_sweep.sh                      # baseline
./scripts/k3_block_sweep.sh PLOW_K3_FUSE_A=1     # any emit knob
python3 scripts/bench/gpuq.py submit glm52-sweep $NGPU ./scripts/glm52_block_sweep_gfx942.sh   # kind-weighted dense/MoE score
```

`k3_block_sweep.sh` takes its own lease (do not wrap it in `gpuq.py submit`). Both scripts pin a
part in their headers (one in its name); re-point them at `$ISA`/`$GPU` before quoting a number. Use the smallest span containing each mixer kind.

### Grid

- Screen at B=1 and the widest compiled rung over short, crossover and long contexts.
- Finalists: B=1/4/16, T=1K–32K, plus points just below/at/above any packet-bucket, KV-ring, tile or
  sliding-window boundary the candidate changes.
- Encode a chosen split count, warp count or register budget in compiler-emitted packet metadata keyed
  by geometry and rung — never a model-name branch or a runtime env knob.
- Reverse control/candidate order, evict L2 symmetrically when production sees different layer
  weights, log SM clock and power beside every timing.
- Close a rung inside out: exact operator → exact unfused/fused sequence → packet role → isolated
  block. Keep packet digest, object hash, input seed, cache state and trial order in the record. A
  candidate that wins alone but loses in its packet role is a rejected search result.

### Anchor and attribute

- Reference floor on `$GPU`: `scripts/block_baseline.py` / `scripts/block_layer_bench.py` time the
  same block through a graph-captured reference path and write the same `sweep.json` schema; join
  with `scripts/block_compare.py --phase {decode,prefill}`. `block_baseline.py --layers N
  --vllm-tpot-ms T` prints the implied floor beside its own number.
- Per-op reference: `python3 scripts/block_op_bench.py <block-config>.json --phases decode,prefill --ctx 1024,4096`.
- Plow per-op on NVIDIA: build the device object with `-DPLOW_NV_TRACE=1`, read the trace summary in `block_run`.
- Roofline (NVIDIA): `scripts/campaign/block_roofline.py <block asset> --ceilings ceilings.json
  --plow sweep.json --current <production sweep.json> [--router-table t.bin] [--vllm layer.json]`.
  Ceilings come from `scripts/bench/calibrate_ceilings.py` (once per box, through gpuq); without them
  the output says NOT calibrated. AMD: `campaign.py block-roofline` (below).

Reading results:

- Correct + at floor → gate passes; name the largest op as Stage 6's first target.
- Correct + slow → report the gap per projection as % of `$BW_BOUND` (decode) or `$COMPUTE_CEIL`
  (prefill), naming the denominator. A uniform gap points at a shared occupancy ceiling.
- Incorrect → stop; localize with the per-substep diff.

## AMD (GLM) block harness

`campaign.py` drives act.x-only AMD blocks. Run inside `nix develop` with render-device access; each
GPU subcommand freezes its inputs, runs the doctor and submits to `scripts/bench/gpuq.py`. A queued
job is not a measurement — check `gpuq.py status` and the job log.

```bash
python3 scripts/campaign/campaign.py block-roofline <recipe>.toml \
  --packet <block.pkt> --program 1 --ctx 8192 --out <scratch>/roofline [--router-table <rank0.act.tab.bin>]
python3 scripts/campaign/campaign.py block-bench <recipe>.toml \
  --packet <block.pkt> --objects <objects> --checkpoint <ckpt> \
  --inputs <operands> --ctx 512 --out <scratch>/block-control [--dstep-log | --trace]
python3 scripts/campaign/campaign.py block-ab <recipe>.toml \
  --control-build <control> --treatment-build <candidate> --inputs <operands> \
  --checkpoint <ckpt> --ctx 512 --out <scratch>/ab --note "one changed lever" [--require-bitwise]
```

- `block-roofline` (CPU only) prices one disassembled decode program with its TP shapes, FP8 scale
  grids and sparse selection width: an optimistic floor with excluded ops listed. `--router-table`
  charges the captured expert union (shape, IDs, gates checked; hash recorded).
- `block-bench`: `--dstep-log` logs host prepare/submit/wait/audit windows (diagnostic; rejected by
  `block-ab`). `--trace` is a separate instrumented run; never score it against uninstrumented
  controls. A failed numerical gate leaves dumps in `outputs/` and no timing report; diagnose with
  `runtime/tests/glm52_real_oracle.py --candidate-dir <dir>` (CPU).
- Objects: `build` accepts gfx recipes with `[objects].script = "scripts/build_gfx950.sh"`, passes the
  emitted `plow_config.h` and records packet/manifest/header/object hashes. `build --object-env K=V`
  is an object-only A/B, recorded apart from emit `--env`.
- `block-ab` keeps the four arms (control/candidate/control/candidate) in one queued lease with the
  CPU-quiet lock; builds cooperate through `scripts/bench/quiets.sh /tmp/plow-campaign-quiet.lock`.
  It refuses mismatched cells/inputs/runtime, instrumented timings, failed numerical gates and
  nonrepeatable same-variant outputs. `--require-bitwise` (scheduling-only levers) demands identical
  outputs; otherwise the captured oracle gates apply. Performance PASS per
  [agent-tools.md](agent-tools.md) "A/B rule"; control/control cannot pass. Results land in
  `comparison.json`; failure exits nonzero. This is a block gate, not serving qualification.
- Routed W8A8 (experimental): `--routed-reference <audit.json>` (a passed pinned routed audit)
  instead of `--require-bitwise`; only the validated reordered/atomic outputs may differ.
- Fixtures: fixed-width decode blocks, capture batch = the packet's decode batch. A fixture holds
  `act.x.bin`, the block's carried `kv.*.bin` / `act.iidx.bin`, `reference.bf16` and
  `reference.json` (`batch`, `ctx`, `tolerance_rel_l2`, stage names); KV files pack
  `[batch, ctx, dimension]`. Export with `runtime/tests/glm52_real_oracle.py --block-inputs <dir>`;
  add `--batch 8 --inputs-only` for distinct seeded sequences; `--block-context <ctx>` with `GLM_L`
  spreads supplied keys over a logical context (checks sparse gathers and RoPE, not indexer choices).
  Prefill needs its own captures. Every row, stage and rank passes before timing; TP counter audits
  stay on.
- Timings are host-clock dispatch/drain, not device kernel time. Block results are never served tok/s.
- Accept PASS only after the queue job exits 0: the lease's final contention audit can still
  invalidate a passing arm.

## Native-kernel inner-loop contract

Applies to every CUDA or HIP body exercised by the block harness. CUDA warps are 32 lanes; CDNA
wavefronts are 64 lanes over 32 four-byte LDS banks (two 32-lane phases) — re-derive launch and
swizzle constants per vendor. Read [`isa/README.md`](isa/README.md) → the `$ISA` file first.

- Inner-loop branches are warp/wave-uniform; profile ragged-boundary predication separately.
- Adjacent lanes issue adjacent, naturally aligned global accesses; a deliberate gather names its
  reuse argument and is measured against coalesced staging.
- Shared/LDS layouts are padded or swizzled against bank conflicts (`ncu` on CUDA; `rocprof` plus the
  LDS instruction census on AMD).
- No new local/scratch traffic: ptxas/SASS spills on CUDA; ISA `scratch_load_*`/`scratch_store_*` on
  AMD (metadata can report zero spill while they exist). Re-check occupancy after any launch-bounds,
  VGPR/AGPR, register or arena change.
- Every `__syncthreads()`/`s_barrier` has a producer-consumer reason; no lane diverges around it.
- Reduce in warp/wave or shared memory first; at most one global atomic per independent warp/wave
  result unless an exact-shape A/B shows otherwise.
- `__restrict__` on non-aliasing pointers; intentional aliasing is stated at the call boundary.
- No device `printf` in production objects; use counter buffers or a separate trace object.

Per packet-derived kernel case: build a hash-distinct object; audit symbols, resources and ISA
(`scripts/asm_audit.py` is the AMD scratch/resource authority); run the deterministic exact-shape
oracle; collect branch, global, LDS/shared, barrier, atomic and occupancy counters; then rotated
control/candidate timing via `scripts/gemma4_h100_kernel_tuner.py`. In its spec bind
`source_contract.device_printf_calls=0`, set `source_contract.pointer_aliasing` to `restrict` or
`intentional-alias`, and put exact-shape ceilings in `anti_pattern_limits`
(`inner_loop_divergent_branches`, `uncoalesced_global_transactions`, `shared_bank_conflicts`,
`barrier_stall_cycles`, `global_atomics`); exceeding one rejects the arm before timing. Record a
rejection at its exact rung; never promote one winner across unrelated M/N/K, attention, live-KV or
packed-topology cells.

## Success criteria

1. Correct within the family tolerance, per substep, not only at the block boundary.
2. No silent zeros (`--inspect`), no NaN/Inf (`block_run check`).
3. Latency at or near the per-block floor measured on `$GPU`; `block_us × layers ≈ decode body`.
4. Every timed run came from a gpuq job that exited 0 (uncontended lease).
5. Emit `--gpu`/`--arch`/`--n-cu` (check `build.json`) and the loaded `$BUILD` object both name
   `$GPU`/`$ISA`; an asset emitted for another part still loads and runs.

## Pitfalls

- Block ms does not scale to model ms (one embed and lm_head in both); a truncated-model score is a ranking.
- An isolated block launches separate norm/RoPE/residual kernels the packet fuses; trust the per-op
  GEMV/attention rows, not whole-block µs.
- An eager reference flatters plow; graph-capture the reference decode step.
- Few-instance ops (router, combine) sink into timer noise on a block; use a wider span or `step_bench`.
- A knob timed on an isolated kernel can invert in the megakernel; rank runtime knobs on a truncated model.
- Example descriptors (`crates/plowc/examples/*.json`) may approximate MoE geometry; fix expert count
  and top-k before comparing MoE.
- Numbers do not transfer between parts, including two parts at the same `$ISA`; re-anchor on `$GPU`.
- The tune system may be a no-op for a `$GPU`/precision (aliased GEMM body, no tunable GEMV); confirm it
  selects something. The real axis there is a `$BUILD` `-D` macro.
- Driver/`$TOOLCHAIN` mismatch (`CUDA_ERROR_INVALID_IMAGE`, "driver too old", HSA equivalent) fails
  runs for reasons unrelated to the block.

## Stop and ask

- A `target.md` row cannot be filled, or no `$GPU` is available.
- The full emit aborts or the block lacks expected tensors (Stage 1–4 defect).
- No C oracle or fixture exists for the family.
- Correctness fails and the per-substep diff does not localize one op.
- The gap is neither uniform nor attributable to one op — hand the per-op table to Stage 6.

## Report

Filled target block; family and block shape and the harness used; correctness (tolerance, per-substep,
silent zeros); the block-vs-roofline table; anchored floor and `block_us × layers` reconciliation;
per-op profile and largest op; gate decision; caveats (anchor measured on `$GPU`, geometry
approximations, `$TOOLCHAIN`/driver).

## Code pointers

| symbol / path | role |
|---|---|
| `devgen::block::{parse_block, write_block_descriptor}` (`crates/devgen/src/block.rs`) | `--block` parsing, `block.json` |
| `glm_emit_block`, `kimi_emit_block`, `nemotron_emit_block`, `glm_build_block[_pf]` (`crates/devgen/src/mla.rs`) | MLA-family blocks |
| `plowrt amd-block`, `disasm`, `simulate` (`crates/plowrt/src/main.rs`) | act.x A/B, static inspection, CPU sim |
| `crates/plowrt/examples/block_run.rs` | NVIDIA `check`, `bench`, `prefill-bench`, `decode-check`, `packed-check`, `mixed-check` |
| `scripts/block_sim.sh`, `scripts/block_e2e.sh` | CPU sim; compile → bench → reference → diff |
| `scripts/block_baseline.py`, `block_layer_bench.py`, `block_op_bench.py`, `block_compare.py` | reference-side harnesses and join |
| `scripts/campaign/block_roofline.py`, `scripts/bench/op_roof.py`, `scripts/bench/calibrate_ceilings.py` | NVIDIA block floor and ceilings |
| `scripts/campaign/nv_block_ab.py` (`campaign.py nv-block-ab`) | NVIDIA four-arm block A/B |
| `runtime/tests/*_block_*_test.c`, `runtime/tests/*_oracle.py`, `scripts/build_glm52_real.sh` | C oracle harnesses |

Related: `docs/arch/13-prefill-chunking.md`, `docs/arch/03-scheduler.md`.
