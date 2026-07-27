# The decode tuner — what it is, how to run it, how to read it

Sweeps the **decode** kernel knobs end-to-end, records the winner per hardware/model/precision
cell, and refuses to publish a number it could not stand behind.

Companion docs: `perf-data/tuner-decode-sweep-design.md` (why it is built this way),
`perf-data/tuner-decode-sweep-h100.md` and `perf-data/tuner-flash-sweep-h100.md` (results),
`perf-data/gemma26b-h100-gemv-mlp.md` (the campaign that motivated it).

> This is the **decode** tuner. `plowc tune --profile prefill_dense` is a separate thing — it
> picks prefill GEMM tiles. The two do not share a search space.

---

## 1. Why it exists

Two facts from the 26B/H100 campaign, both measured:

**Defaults rot.** `GV_MOE_UN` was optimal at 4, then the arms around it improved and 2 became
optimal (6.288 → 6.194 ms). `PLOW_NS_ABS` was 8, then the flash rewrite landed and 16/32 became
optimal. Nobody changed those constants; the code around them moved.

**Optima are not constants — they invert with occupancy.** `GV_UNROLL` 8 wins at 1 block/SM and
loses at 2 (the register cap is 128 there and deep unroll spills). `PLOW_NV_FP8_RB` 4 wins at
occ-1, 2 at occ-2. `PLOW_NS_ABS` 16 at occ-1, 32 at occ-2 — two clean U-shapes with different
minima. A single `#define` cannot serve both.

**…and they invert with BATCH.** `GV_MM_MAX` is the widest `gemv_*_rows<MM>` instantiated, so a
batch of B costs `ceil(B/GV_MM_MAX)` weight passes. Measured end-to-end on the 5090
(`perf-data/px15-tunedb-sm120.md`): at B=1 the two arms are 0.15 % apart, at B=8 `=8` beats `=16`
by **34.8 %**. `batch` is therefore part of `DecodeCell`, and it is a **packet** axis — the decode
batch is `PLOW_DECODE_BATCH` at emit time, not `plowc --batch` (prefill buckets) and not a
`step_bench` argument, which only ever *clamps* to what the packet already has.

---

## 2. The one rule that shapes everything: score end-to-end

**The isolated microbench disagrees with the megakernel.** `runtime/nvidia/experiments/gemv_lab_h100.cu`
says row-blocking wins 1.4× on *every* decode shape. In the real interpreter it **loses**. A
microbench-scored tuner would confidently ship the wrong arm.

So: the lab **prunes**, `step_bench` TPOT **scores**. There is deliberately no field in
`DecodeMeasurement` for a microbench number.

---

## 3. Run it

```bash
# build once
nix develop -c cargo build --release -p plowc
nix develop -c cargo build --release -p plowrt --features cuda --example step_bench
nix develop -c cargo build --release -p tunedb --bin tunedb-decode

scripts/tune_decode_sweep.sh \
  --model /workspace/models/gemma-4-26B-A4B-it \
  --dtype fp8 \
  --ctx 1024,8192 \
  --out perf-data/tune-decode-h100-26b-fp8.jsonl
```

Every GPU command goes through `gpulease`, and every plow binary needs
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`. The script does both.

**Pick fp8 when memory is tight.** A 26B **bf16** configuration needs ~53 GiB. If anything else
holds the card you get `CUDA_ERROR_OUT_OF_MEMORY` and the sweep cannot start. fp8 needs ~30 GiB
and fits. That is arithmetic, not a preference.

### Ablation twins — measure an op, not the step

`--ablate-lo/--ablate-hi` builds a second object per config with one opcode's **body** compiled
out while every gate and signal stays. Each row then carries `median_ms`, `ablated_ms`, and
`op_cost_ms = median - ablated`.

```bash
# opcode 12 = FLASH_DECODE (13 = FLASH_MERGE, 10 = GEMV, 30 = GEMV_FP8, 71 = MoE GLU)
scripts/tune_decode_sweep.sh ... --ablate-lo $((1<<12))
```

Use it. It changed the conclusion three times in one round:
- `FA_GF_FULL=8`'s regression is only ⅔ flash — the rest is a **constant arena tax**, because
  the dynamic-smem arena is a *union sized by the largest claim* and that claim is flash's, so
  widening it bills every other op.
- `FA_WPR=1` cuts flash 2.53× with the non-flash remainder **unchanged** — proof it moved only
  the arm it names, rather than an assumption.
- `NS_FULL_ABS` trades flash-body time against gate time; net sign flips with ctx.

On total TPOT alone all three would have been one opaque number and two would have been
attributed to the wrong arm.

---

## 4. Read and use the results

```bash
tunedb-decode ingest --results perf-data/tune-decode-h100-26b-fp8.jsonl
tunedb-decode best   --hardware nvidia/sm_90a/h100-nvl --model gemma-4-26B-A4B-it \
                     --dtype fp8 --n-cu 264 --ctx 1024
```

`best` prints the winning knob set plus the flags that rebuild it:

```bash
PLOW_EXTRA_DEFINES="$(tunedb-decode best ... --print defines)" \
  scripts/build_sm90a_cubin.sh out/interp_sm90a.cubin
```

Packet-side knobs (`PLOW_NS_ABS`, `PLOW_NS_FULL_ABS`, `PLOW_FA_GF_FULL`, `--n-cu`) come back from
`--print emit` and go to `plowc`. **They are deliberately not the same list** — object flags and
packet flags land in different artifacts and drift apart exactly when written down as one string.
A cubin built for 2 blocks/SM against a 132-block packet is not slower, it is a launch the engine
refuses.

`--print` refuses when the filter leaves more than one cell standing: a flag string names ONE
object, and the union of two cells' winners is an object nobody measured. Narrow with
`--model / --dtype / --n-cu / --batch / --ctx`.

### Knobs that are PAIRS, and must be swept as one axis

Three so far. Setting half of a pair does not measure the knob, it measures the disagreement:

| pair | object half | packet half | what breaks |
|---|---|---|---|
| occupancy | `PLOW_NV_FORCE_MINBLK` | `--n-cu` | the engine refuses the launch (loud) |
| full-layer GQA fusion | `PLOW_NV_FA_GF_FULL` | `PLOW_FA_GF_FULL` | the emitter sizes `nsplit` from `n_grp = heads/GF`; a mismatch silently mis-fills the grid |
| decode batch | — | `PLOW_DECODE_BATCH` | `step_bench` clamps to the packet's batch **silently**, filing a B=1 timing under B=8 |

`DecodeKnobs::defines()` and `emit_env()` render both halves of each, so a record cannot store
one without the other.

### Provenance: why your rows may say `provisional`

`ingest` **refuses to qualify** a row that could not verify an idle GPU, and names the worst
resident reading. `tuning/README.md` stated that policy; nothing enforced it, and `gpulease`'s
audit is namespace-blind on this box, so `rc=0` genuinely does not mean the card was ours.

Provisional rows are **stored but unselectable**. To publish, re-run on a verifiably idle card.
Do not "fix" this by passing `--provisional` and quoting the number anyway: a tight rep spread
argues the timings are internally consistent, it does not verify the card was yours.

---

## 5. Extending it

### A new knob in a family that already exists
Add a typed field to `DecodeKnobs`, render it in `defines()` (object) or `emit_env()` (packet),
add it to `label()`. Use `Option<u32>` when "not overridden" differs from a value — the shipped
recipe sets `FA_WPR=1` while the source defaults to `0`, so recording `0` for unset would
describe an object nobody built.

### A whole new op family
Use `extra_defines` / `extra_emit` (`BTreeMap<String,String>`) instead of growing the struct.
Both are `serde(default)`, so **older rows still load and there is no schema break**. This is
the supported path; it exists because the flash family arrived after the struct was written and
its data had nowhere to go.

### Ranking on blocks, scoring on the model
`scripts/tune_decode_block_sweep.sh` sweeps a knob on a single-layer block asset
(`plowc --block N` + `examples/block_run`) in seconds instead of minutes. This is **not** the
`gemv_lab` mistake §2 forbids: a block drives the REAL interpreter — same cubin, same dispatch,
same counter protocol, same register footprint — and what made the standalone bench lie was that
none of that was present. It still cannot give a magnitude, only an ordering, so the protocol is:
rank wide on blocks → confirm the shortlist end-to-end → publish only the confirmed number.

Measured agreement (px15, `GV_MM_MAX` at B=8): block −32.3 %, model −34.8 %, and an independent
campaign −33.8 %. Record any case where the two disagree — that is `gemv_lab` reappearing and it
is the most valuable thing a block sweep can tell you.

### A new GPU
`DecodeCell.hardware` is already a `HardwareFingerprint::tuning_path` string
(`nvidia/sm_90a/h100-nvl`), so records never collide across parts. For a new **vendor**, add a
`Backend` variant and teach `defines_for` its flag syntax. It returns `None` for a backend with
no sweep rather than emitting nvcc `-D` spellings that would build the wrong object — so a
missing backend fails loudly instead of silently producing a wrong artifact.

Tests covering all three: `a_new_op_family_rides_the_extra_maps`,
`knobs_without_extras_still_load`, `a_backend_without_a_sweep_refuses_to_render_flags`.

---

## 6. Practical notes

- **Cost.** ~40 s to build an object, ~25 s per `step_bench` point, ×reps ×ctx. A 32-config ×
  4-ctx grid is ~75 min per (gpu, dtype). It belongs in `tunedb` as a recorded artifact;
  `compile` reads it and must never write it, or a build calibrates against its own output.
- **Long-ctx is expensive for a bad reason.** fp8 packets carry no prefill program, so a
  ctx=32768 run spends ~165 s consuming the prompt one decode step at a time (vs ~20 s at 1k).
  **Emit packets with prefill buckets before any long-ctx sweep** — cheapest available speedup.
- **When is flash worth tuning?** Measured: flash is **7.8 %** of the step at ctx=1024 and
  **23.7 %** at 32768, while the non-flash remainder is flat to 0.007 ms across that 32× range.
  Below ~8k, tune GEMM knobs; above it, tune flash.
- **Objects are cached by the SHA of their full define string**, so re-running a grid re-measures
  without rebuilding, and an unnamed axis emits no `-D` at all — a sweep that ignores a family
  builds objects byte-identical to ones from before that family existed.
- **sm_120 must stay byte-identical.** Every knob defaults to its current source value. Verify by
  building `scripts/build_sm120_cubin.sh` from a clean `git archive HEAD` and from your tree and
  comparing sha256 of **both** the decode and `_pf` cubins.
