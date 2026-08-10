# The gfx942 recipe: how to build, measure and gate on this box — and what to measure next

> **Scope:** 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **METHOD** —
> the operating manual for this hardware. Every command here has been run; every gate here has
> caught a real failure. Results live in `glm52-experiments.md`, method lessons in `LESSONS.md`.

This exists because the campaign kept paying the same tolls twice. Each rule below cost hours
the first time.

---

## 1. THE FIVE GATES, in the order they must fire

A number that skips any of these has been wrong before, in this directory, with a plausible
value.

| # | gate | what it catches | cost |
|---|---|---|---|
| 0 | **binary carries HSA** — `grep -aq libhsa-runtime64 target/release/plowrt` | `cargo build`/`test` without `--features hsa` relinks plowrt into a CPU-only binary that serves CORRECT answers at fictional speed | ~1 ms |
| 1 | **GPU lock + no sibling plowrt** (`pgrep '^plowrt'`) | a sibling run makes every timing fiction; a `kill -9` leaves the persistent megakernel resident | seconds |
| 2 | **HSA backend selected** (log has no `CPU reference backend active`) | the same fallback as gate 0, reached via a missing `LD_LIBRARY_PATH` *inside* nix | 75 s (the load) |
| 3 | **coherence** ("capital of France" → Paris) | a fast wrong server | seconds |
| 4 | **accuracy, PAIRED, full set** — `gsm_paired.py` + `mcnemar.py` | a numerics change that degrades answers. **`run_plow.sh`'s n=100 aggregate CANNOT do this**: measured 0.970 vs 0.950 = 0.72 sigma, McNemar p ~= 0.50 | 73 min/arm |

Gates 0–3 are automatic in `scripts/twoengine/run_plow.sh` and `run_gsm_paired.sh`. Gate 4 is
only needed when an arm changes numerics.

---

## 2. Build

**Objects (OUTSIDE nix** — hipcc is the system ROCm one and nix's `CPATH`/`LIBRARY_PATH` shadow
the glibc it was built against):

```sh
PLOW_HIPCC=/opt/rocm-7.2.4/bin/hipcc HIP_PATH=/opt/rocm-7.2.4 \
ROCM_PATH=/opt/rocm-7.2.4 ROCM_HOME=/opt/rocm-7.2.4 \
PLOW_OCC4=1 PLOW_L2HIER=1 JOBS=14 bash scripts/build_gfx942.sh <outdir>
```

* `/opt/rocm/bin/hipcc` on this box is **broken** (its internal `clang++` is missing) — hence
  `PLOW_HIPCC`.
* `PLOW_ROWS_ONLY=interp_flash` builds one family for iteration; the result is PARTIAL and also
  skips `test_kernels.elf`.
* Default-ON for gfx942 as of 2026-08-10: `PLOW_GEMV_LG`, `PLOW_MLA_PF_SV`, `PLOW_MOE_PF_EPI`,
  `PLOW_MOE_PF_DET`, `PLOW_L2HIER`, `PLOW_MLA_PF2_DBUF`, `PLOW_MOE_DEC_LG`,
  `PLOW_MLA_FOLD_TB=8` (TTFT −3.8/−5.5/−6.2% @4k/8k/16k), `PLOW_XR_AGG` (TTFT −1.7% @4k),
  and the `PLOW_GLM_FUSE_QNORM` decode-object ARM (the emit fold stays opt-in).
  `FOLD_TB`/`XR_AGG` were reverted 08-09 on a needle failure and re-adopted 08-10:
  XR_AGG's ordering was really broken (fixed in `op_collective.h` — system-scope release
  arrival RMW + system acquire in the closer; the old failure was intermittent), FOLD_TB
  was never solo-gated and passes 4/4 alone. Record: LESSONS.md #19, glm52-batched-decode-r4.md §r7.

**plowrt — the feature flag is load-bearing:**

```sh
nix develop . -c cargo build --release -p plowrt --features hsa
```

Any other cargo invocation in the tree (`cargo test --workspace`, `cargo build --bin plowc`)
**relinks `target/release/plowrt` without HSA**. Gate 0 exists for this.

**Emit** (GLM-5.2 TP8, current best — the COMPLETE environment; a command that omits any of
these emits a measurably slower or differently-shaped blob and was this file's own P1 defect):

```sh
GLM_FULL=1 PLOW_MLA_PREFILL=full PLOW_MLA_PF_V2=1 PLOW_GLM_PF_NS=2 PLOW_GLM_DSA=0 \
GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 \
PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1 PLOW_GLM_FUSE_B1=1 PLOW_GLM_FUSE_QNORM=1 \
PLOW_MOE_PF_DET=1 \
plowc --emit devblob --hf-dir /workspace/models/GLM-5.2-FP8 --gpu MI300X --arch gfx942 \
      --num-gpus 8 --max-ctx 73728 --out <assetdir>
```

* `PLOW_MLA_PF_V2=1` **at emit** (not only at serve): `packet::devbuild` reads it to split
  `FlashMlaPrefill` into its own wave-class-4 segment — without it MLA prefill never reaches
  the flash object and every flash arm reads as a structural null.
* `PLOW_GLM_DSA=0` is the shipped best (dense wins at every ctx to 32k) and is REQUIRED with
  `FUSE_QNORM` at max-ctx > 65536 (the armed indexer is a second consumer of `n.qlat`).
* The folds: `FUSE_ROPE` (byte-identical, −0.26 ms/tok), `FUSE_SEAM` (−0.37 ms), `FUSE_B1`,
  `FUSE_QNORM` (−1.27% TPOT; needs the armed decode object, default-on as of 2026-08-09),
  `MOE_PF_DET` (TTFT −2.07% @8k; object arm default-on).
* **Batched-decode ladder** (throughput serving): add `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16`
  and drop `PLOW_GLM_FUSE_QNORM` (the fold's M>1 staging is unvalidated — the emit refuses).
  The OBJECTS must be built `PLOW_DECODE_BATCH=16` (widens `PLOW_GEMV_MM` and compiles the
  grouped-`*Pf` arms into the decode row). The `rows==1` rung re-emits byte-identical to the
  unladdered blob — that anchor is checked, not hoped.

**The asset's `checkpoint/` must be the PREPPED weight dir, not the raw HF checkpoint.**
`plowc --emit` leaves `<assetdir>/checkpoint` pointing at `--hf-dir`, but the blob binds
derived/dequantized names (`glm52_prep*.py`: absorbed `Wqa`/`Wuv`, bf16 `q_a_proj`/`o_proj`/
shared expert) that only the prep dir carries — on this box
`/workspace/models/GLM-5.2-plow-lite`. Point the symlink there before serving. A raw-HF
checkpoint fails two ways, one loud (`shard ... replicated but the checkpoint has N B and the
blob declares 2N B`), one NOT: with `FUSE_QNORM` the mis-sized `q_a_proj` mapping is read past
its end by the fold's staged GEMV and dies as a **GPU memory access fault at load** — that
fault cost this campaign half a day of bisecting object knobs that were never the problem.

**Serve** — `PLOW_MLA_PF_V2=1` is required (the blob carries the causal KV-split `ns=2`; without
it the load is REFUSED, loudly), and `LD_LIBRARY_PATH` must be set **inside** the nix shell.
`scripts/glm52_serve_smoke.sh <assets> <port>` is the load-fault smoke: readiness + coherence
and a clean teardown (TERM → drain → KILL — a bare `kill -9` leaves the megakernel resident
and wedges every later test on the box).

---

## 3. TIER THE EXPERIMENT. Do not run the whole network to answer a kernel question.

The campaign's own evidence for why (`tune_block_sweep.sh` header): an isolated kernel
microbench said row-blocking wins 1.4x and it LOST in context; a GEMV harness timed one
`gemv_rows<16>` as equal to two `gemv_rows<8>` while in the megakernel the same knob was worth
41.17 -> 28.8 ms. A **single-layer block asset runs the real megakernel**, so it keeps the
context, and it reproduced the full-model ratio to **1.4%** at **~1/15th the cost**.

| tier | instrument | answers | cost |
|---|---|---|---|
| 0 | `hipcc -S` + `probes/asm_loops.py` | "did the instruction I wanted get emitted" | seconds, no GPU |
| 1 | isolated kernel bench (`runtime/bench/amd/glm52_*`) | "is this kernel faster in isolation" — **does not predict the megakernel** | ~1 min |
| 2 | **single-layer block, context sweep** (`plowc --block L`) | "does it help in the real megakernel" — the workhorse | ~3 s/run after emit |
| 3 | full model, `run_plow.sh` | the headline TTFT/TPOT vs vLLM | ~8 min/arm |
| 4 | full model + `gsm_paired.py` | does it change ANSWERS | ~73 min/arm |

**Tier 0 licenses a measurement, never an adoption.** Branchless `f2bf` was −5.0% instructions
and +5.3% wall clock (LESSONS 17). `d16_hi` disassembled perfectly and computed the wrong answer
(LESSONS 14).

### Tier 2 for GLM-5.2 specifically

GLM is **3 dense + 75 MoE** layers, so a block score is the kind-weighted sum of MARGINAL
per-layer cost:

```
score = 3 * L_dense + 75 * L_moe
```

Time ONE block of each kind (`--block 0` dense, `--block 3` MoE — `first_k_dense_replace = 3`).
A 1-layer block also carries the embedding/lm_head declaration as a fixed overhead `O`; difference
a 2-layer against a 1-layer block to get it. `O` is constant across knobs, so it cancels in a
RANKING — treat the score as a comparator, not a predicted TTFT.

**`--max-ctx` is load-bearing at tier 2**: the default arms the DSA indexer (`GlmCfg::dsa` gates
at `ctx > 65536`), so use `--max-ctx 4096` unless DSA is the thing under test.

---

## 4. Instruments that already exist — check before building

| tool | for |
|---|---|
| `scripts/twoengine/run_plow.sh` | tier 3, gates 0–3 + TTFT ladder + TPOT + GSM8K(n) |
| `scripts/twoengine/run_gsm_paired.sh` + `mcnemar.py` | gate 4, the only instrument that can resolve a 2 pp accuracy move |
| `scripts/rebench_tune_gemm_gfx942.sh` | re-run the GEMM tile campaign (see §5) |
| `perf-data/plow-gfx942/probes/asm_loops.py` | loop extraction + instruction mix from a disassembly |
| `probes/d16sem.hip`, `probes/f2bf_gate.c` | templates: prove an instruction's semantics / prove value-identity exhaustively |
| `scripts/glm52_prefill_gate.sh` | per-stage residual vs an HF oracle, single block |
| `plowc tune status --gpu MI300X` | **is this build's tile selection measured or analytical** |

---

## 5. THE TUNING STORE WILL GO STALE, AND IT IS SILENT

The store is keyed on `defines + toolchain + preprocessed-source digest`
(`kernelcaps::BuildId::label`). **One edit under `runtime/amd/` — or one flipped default in
`build_gfx942.sh` — re-stales every record at once**, after which `pick_tile` reverts to the
analytical model and reports tier `portable`, which is byte-identical to what it reports when
nothing was ever measured.

On 2026-08-09 every record in BOTH AMD cells was stale (gfx942 196, gfx950 3080) and had been
for days, through every published measurement.

```sh
plowc tune status --gpu MI300X          # the check — reads the digest census
scripts/rebench_tune_gemm_gfx942.sh     # the fix — rebuild, measure, ingest, verify
```

Blobs now record their own provenance: `build.json` → `tuning.tile_source` is
`measured` | `mixed` | `analytical`. **Check it before quoting a number.**

---

## 6. What to measure next, in expected-value order

The 8k gap to vLLM is ~950 ms and the 16k gap ~1900 ms. Attribution (which itself needs
re-deriving once tiles are measured — see below):

| term | 8k | 16k | state |
|---|---:|---:|---|
| MoE grouped | 515 | 1007 | 2.25x off aiter; the `part` round-trip fix is ADOPTED (`PLOW_MOE_PF_DET`) |
| attention (flash + merge) | 338 | 1088 | the majority at 16k; `d16_hi` and softmax-VALU both refuted |
| attn+shared GEMM | 188 | 378 | **0.51x hipBLASLt — and currently tile-selected by the ANALYTICAL MODEL** |
| collectives | 163 | 368 | at a proven floor |
| schedule overhead | 124 | 282 | at a proven floor (91% packed at 8k) |

1. **Re-run the gfx942 GEMM tile campaign** (§5). Cheapest, and it *gates the ranking below* —
   part of the 0.51x may be selection error rather than kernel rate, which would move GEMM work
   up several places. Nothing else should be re-prioritised until this is known.
2. **Adopt the emit-side folds** — `FUSE_ROPE` (byte-identical) + `FUSE_SEAM` = −2.7% TPOT,
   already measured, sitting behind hand-passed env vars.
3. **fp8-latent KV, re-measured** (task #39). Rejected at +0.4…+3.1% TTFT *before* the 6-VALU
   dequant fix that the audit explicitly queued to compose with it. Second-order prize: it halves
   `rl[]` and is the only remaining route to the registers a depth-2 flash DBUF needs.
4. **MoE grouped rate** beyond the combine fusion — 6.3x off plow's OWN roofline at 8k.
5. **Batched decode** (task #24) — throughput is pinned at 22.8–27.0 tok/s from concurrency 1 to
   32. Blocker is precise: the MoE DECODE ops carry no token dimension; route is to emit the MoE
   seam with the PREFILL op family at `T = rows`.
6. **TP8 DSA decode crossover** (task #41) — the constant is a TP4 calibration and we serve TP8.

**Do NOT re-derive:** DSA sparse prefill (closed at audit grade, model-property ceiling),
per-XCD prefill placement (+97.6% at 16k), the GEMM tile ladder re-cut (+11.7%), `d16_hi`
(refuted in hardware), branchless `f2bf` (refuted served), collectives (correctly tuned),
packet-boundary protocol (null on two instruments).
