#!/usr/bin/env bash
# glm52_block_sweep_gfx942.sh — TIER 2 for GLM-5.2 on gfx942: a FEW-LAYER TP8 block, swept over
# PREFILL context, scored as the kind-weighted marginal per-layer cost.  [RECIPE-gfx942 §3]
#
# WHY THIS TIER EXISTS. Answering a kernel question with `run_plow.sh` costs a 75 s model load plus
# minutes per arm, and an isolated microbench answers a different question: the campaign's own
# evidence is in `scripts/tune_block_sweep.sh` (a GEMV harness timed one gemv_rows<16> as equal to
# two gemv_rows<8> while in the megakernel the same knob was 41.17 -> 28.8 ms). A truncated MODEL
# blob runs the REAL megakernel at the REAL TP degree, so it keeps the context an isolated kernel
# loses, and it costs ~1/10th of a tier-3 arm.
#
# WHY A TRUNCATED MODEL AND NOT `plowc --block L`. On the Gemma/NVIDIA path a `--block` asset still
# declares embed/lm_head and `examples/block_run` drives it. **The GLM `--block` path does not**:
# `glm_build_block_pf` emits act.x-in/act.x-out programs with no Embed, no lm_head and no
# act.logits (crates/devgen/src/mla.rs), so `plowrt amd-bench` cannot drive one — the only runner is
# the TP1 C harness `runtime/tests/glm52_run.c`, which needs an HF oracle fixture and would measure
# a TP1 geometry we do not serve. `PLOW_LAYERS=N` gives the same thing a block gives (a few real
# layers of the real megakernel) at the TP8 geometry that IS served, and it differences the fixed
# overhead out exactly.
#
# THE SCORE. GLM-5.2 is 3 dense + 75 MoE (`first_k_dense_replace = 3`), and the two kinds are NOT
# interchangeable — a dense layer routes the SAME grouped expert arms under degenerate 1-expert
# routing, at 1/256th the expert traffic. So we time three spans and difference them:
#
#     L_dense = ( T(N_DENSE) - T(N_LO) ) / (N_DENSE - N_LO)      layers 1..3 are all dense
#     L_moe   = ( T(N_MOE)   - T(N_DENSE) ) / (N_MOE - N_DENSE)  layers 3.. are all MoE
#     score   = 3 * L_dense + 75 * L_moe
#
# **THE SCORE IS A COMPARATOR, NOT A PREDICTED TTFT.** It omits everything that does not scale with
# the layer chain — the embed, the lm_head, the sampler, the host-side chunk plan, the one-time
# buffer touches — because differencing removes them. That fixed term `O` is printed (as
# `T(N_LO) - L_dense`) precisely so nobody mistakes the score for a wall clock. `O` is constant
# across knobs, so it cancels in a RANKING, which is the only claim this instrument makes.
#
# WHY SPANS AND NOT ONE LAYER. One MoE layer at ctx 4096 is ~12 ms and the effects worth ranking are
# ~1%. Eight layers put the difference at ~100 ms with the same absolute timer noise, i.e. 8x the
# resolution for ~1.6x the run cost (the cost is dominated by the LOAD, not by the prefills).
#
# `--max-ctx 4096` IS LOAD-BEARING: the default arms the DSA indexer (`GlmCfg::dsa` gates at
# ctx > 65536), which is unvalidated past 2048. Every context in the sweep is also a real compiled
# bucket rung (glm_prefill_buckets = 128,512,1024,2048,4096 at this max-ctx), so no rung is measured
# through padding.
#
# WHAT THIS TIER CANNOT SEE, stated up front:
#   * Anything that only engages at T >= 2048. `derive_segments` routes FlashMlaPrefill to the
#     4-wave flash object only when `PLOW_MLA_PF_V2=1 && prog.t >= 2048`, so EVERY flash-object knob
#     (PLOW_MLA_PF_SV, PLOW_MLA_PF2_DBUF, ...) is a structural NULL at ctx 512/1024. A flat 512/1024
#     row is the harness working, not the harness failing.
#   * Anything with few instances per span (see k3_block_sweep.sh's caveat 2).
#   * Register/occupancy differences if the truncated blob instantiates a different arm set.
#
# THE CALIBRATION, and it is the only reason to believe any of the above. `PLOW_MLA_PF_SV` is an
# object-only, bit-identical flash-object arm whose FULL-MODEL effect is on record (three
# interleaved rounds, `glm52-flash-streamed-v.md`): TTFT -1.2% @4k / -2.5% @8k / -2.5% @16k.
# Run here as `sv0` vs `sv1` (identical object dirs except the four `interp_flash*.elf`, which
# differ only in `-DPLOW_MLA_PF_SV=1`), with each arm DUPLICATED to measure the harness's own
# round-to-round spread:
#
#   ctx    control (sv0 vs sv0b)     sv1      sv1b     effect    tier-2 / full-model TTFT
#   2048   -0.22%                   -1.94%   -2.34%   -2.14%    (no full-model rung)
#   4096   -0.47%                   -3.22%   -3.06%   -3.14%    -1.2%
#   8192   -0.09%                   -4.62%   -4.74%   -4.68%    -2.5%
#    512   +0.32%                   -0.48%   -0.22%    NULL     (below the T>=2048 flash routing)
#   1024   +0.30%                   -0.04%   +0.71%    NULL     (idem)
#
# DIRECTION: correct at every context where the arm can engage, at 5-50x the control spread, with
# both SV rounds below both control rounds. SCALING: correct — the win grows with KV-tile count,
# which is what an LDS-side fix predicts. NULL WHERE IT MUST BE NULL: flat at 512/1024, where
# `derive_segments` does not route MLA prefill to the flash object at all.
#
# MAGNITUDE: the score moves ~2x further than full-model TTFT does, and it should. The score is the
# LAYER CHAIN ONLY; TTFT also contains `O` and everything else outside the chain. Multiplying by
# the layer-chain fraction of TTFT (score/TTFT = 484/973 = 0.50 @4k, 914/1677 = 0.545 @8k, using
# this campaign's published TTFT) converts one to the other:
#   @4k  -3.14% x 0.50  = -1.57%  vs -1.2% measured
#   @8k  -4.68% x 0.545 = -2.55%  vs -2.5% measured
# Use the score to RANK. If a predicted TTFT delta is wanted, scale it by that fraction and say so.
#
# L_dense IS THE NOISY HALF (+-2.5% round to round): it is a 2-layer difference on the smallest
# quantity in the table. It carries 3/78 of the score, so it rarely matters — but do not rank a
# dense-only knob on it without more reps.
#
#   scripts/glm52_block_sweep_gfx942.sh <matrix-file> [out.tsv]
#
# MATRIX FILE: one arm per line, `name<TAB>/path/to/hsaco-dir<TAB>RUNTIME_ENV=1 ...`.
# Blank lines and `#` comments ignored; a `base` row is a good idea. Column 2 is the OBJECT axis
# (build the dirs with `scripts/build_gfx942.sh`, which is where the -D knobs live on this box);
# column 3 is the RUNTIME axis (PLOW_* the engine reads). `PLOW_MLA_PF_V2=1` is added to every arm —
# the blob carries the causal KV-split (ns=2) and the load is REFUSED without it.
#
# EMIT-SIDE arms are deliberately NOT a matrix column: they change the blob, so they need their own
# assets. Run the script twice with different `GLM52_EMIT_ENV` and `WORK`.
#
# GATES, in the order RECIPE-gfx942 §1 requires:
#   0  binary carries HSA          — a cargo build without `--features hsa` serves correct answers
#                                    at fictional speed; caught in ~1 ms, before the lock.
#   1  GPU lock + no sibling plowrt
#   2  HSA backend + gfx942        — checked per run out of the engine's own banner.
#   +  OBJECT MANIFEST             — the objects the engine actually OPENED are printed. An arm
#                                    built into an object the run never loads reports a confident
#                                    null (RECIPE §"OBJECT-SELECTION TRAP"); this is the check.
# Gates 3/4 (coherence, accuracy) do not apply: weights are UNBOUND here on purpose, so the tokens
# are meaningless and the timing is not. NOTHING numeric may be read out of a run using this.
set -euo pipefail

MATRIX="${1:?usage: glm52_block_sweep_gfx942.sh <matrix-file> [out.tsv]}"
OUT="${2:-glm52-block-sweep.tsv}"
ROOT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="${WORK:-/tmp/glm52-block-sweep}"

CKPT="${GLM_CKPT:-/workspace/models/GLM-5.2-FP8}"
CTXS="${CTXS:-512,1024,2048,4096}"
REPS="${REPS:-3}"
TP="${TP:-8}"
MAXCTX="${MAXCTX:-4096}"

# The three spans. N_LO..N_DENSE must stay inside the dense prefix (first_k_dense_replace = 3);
# N_DENSE..N_MOE must stay outside it. The model's own kind counts weight the score.
N_LO="${N_LO:-1}"
N_DENSE="${N_DENSE:-3}"
N_MOE="${N_MOE:-11}"
MODEL_DENSE="${MODEL_DENSE:-3}"
MODEL_MOE="${MODEL_MOE:-75}"

# The blob recipe. This is the shipped prefill configuration (RECIPE §2) at a 4096 max-ctx.
#
# `PLOW_MLA_PF_V2=1` IS AN EMIT-SIDE KNOB AND THIS COST THE HARNESS ITS FIRST ANSWER. It is
# documented as a SERVE flag ("serve env still needs PLOW_MLA_PF_V2=1"), but `packet::devbuild`
# reads it too (devbuild.rs: `let mla_v2 = !uniseg && env PLOW_MLA_PF_V2 == "1"`) and that read is
# what SPLITS `FlashMlaPrefill` into its own wave-class-4 segment. `exec::amd::derive_segments`
# then routes a segment to the 4-wave flash object only if EVERY entry in it is FlashMlaPrefill —
# so a blob emitted WITHOUT the flag has impure segments, MLA prefill runs on the 8-wave prefill
# object, and every flash-object knob (PLOW_MLA_PF_SV, PLOW_MLA_PF2_DBUF, ...) is a STRUCTURAL
# NULL at every context. Measured here: without it, SV moved the score by -0.12/-0.20/+0.43/-0.10%
# at 512/1024/2048/4096 — noise — against a known -1.2% full-model effect.
EMIT_ENV="${GLM52_EMIT_ENV:-PLOW_MLA_PREFILL=full PLOW_MLA_PF_V2=1 GLM_MOE_CORESIDENT=2 GLM_SHARED_CUS=48 GLM_SHARD_HEAD=1 PLOW_GLM_PF_NS=2 PLOW_GLM_FUSE_ROPE=1 PLOW_GLM_FUSE_SEAM=1}"
case " $EMIT_ENV " in *" PLOW_MLA_PF_V2=1 "*) ;; *)
  echo "FAIL: GLM52_EMIT_ENV has no PLOW_MLA_PF_V2=1. Without it the emitted prefill segments are"
  echo "      impure, MLA prefill never reaches the 4-wave flash object, and every flash-object"
  echo "      arm reads as a null. Add it, or set GLM52_ALLOW_NO_V2=1 if that is the thing"
  echo "      under test."
  [ "${GLM52_ALLOW_NO_V2:-0}" = 1 ] || exit 1;; esac

NIX="${PLOW_NIX:-/nix/var/nix/profiles/default/bin/nix}"
ROCM_LIB="${PLOW_ROCM_LIB:-/opt/rocm-7.2.4/lib}"
PLOWRT="${PLOWRT_BIN:-$ROOT/target/release/plowrt}"
PLOWC="${PLOWC_BIN:-$ROOT/target/release/plowc}"
LOCK="${PLOW_GPU_LOCK:-/tmp/plow_gpu.lock}"
mkdir -p "$WORK"

# ---- GATE 0: the binary carries HSA -------------------------------------------------------
# `cargo build`/`cargo test` WITHOUT `--features hsa` relinks plowrt into a CPU-only binary that
# runs the schedule CORRECTLY at a fictional speed. One millisecond here, before the lock.
[ -x "$PLOWRT" ] || { echo "FAIL: no plowrt at $PLOWRT"; exit 1; }
grep -aq "libhsa-runtime64" "$PLOWRT" || {
  echo "FAIL: $PLOWRT was built WITHOUT --features hsa (no libhsa-runtime64 reference)."
  echo "      Rebuild: nix develop . -c cargo build --release -p plowrt --features hsa"; exit 1; }
"$PLOWRT" amd-probe --help 2>/dev/null | grep -q -- --prefill-sweep || {
  echo "FAIL: this plowrt has no 'amd-probe --prefill-sweep'. That flag is what lets ONE load"
  echo "      cover every context with a warmed median; without it the harness would spend 99%"
  echo "      of its wall clock loading. Rebuild plowrt from this tree."; exit 1; }
[ -x "$PLOWC" ] || { echo "FAIL: no plowc at $PLOWC"; exit 1; }
[ -e "$ROCM_LIB/libhsa-runtime64.so.1" ] || {
  echo "FAIL: no libhsa-runtime64.so.1 under '$ROCM_LIB' (set PLOW_ROCM_LIB)"; exit 1; }
[ -x "$NIX" ] || { echo "FAIL: nix not found at '$NIX' (set PLOW_NIX)"; exit 1; }

# ---- the matrix ---------------------------------------------------------------------------
# Split on tabs MANUALLY: `IFS=$'\t' read` collapses consecutive tabs, so a row with an empty
# column silently shifts the next one into it (the trap tune_block_sweep.sh records).
names=(); objs=(); envs=()
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue;; esac
  n="${line%%$'\t'*}"
  rest="${line#*$'\t'}"; [ "$rest" = "$line" ] && rest=""
  o="${rest%%$'\t'*}"
  e="${rest#*$'\t'}"; [ "$e" = "$rest" ] && e=""
  [ -f "$o/interp_prefill_fp8_mla_moe_gq.elf" ] || {
    echo "FAIL: arm '$n': $o has no interp_prefill_fp8_mla_moe_gq.elf — is it an object dir?"; exit 1; }
  names+=("$n"); objs+=("$o"); envs+=("$e")
done < "$MATRIX"
[ "${#names[@]}" -gt 0 ] || { echo "FAIL: matrix $MATRIX has no arms"; exit 1; }
echo "sweep: ${#names[@]} arm(s), spans N=$N_LO/$N_DENSE/$N_MOE, ctx {$CTXS}, ${REPS} timed rep(s)"

# ---- phase 1: EMIT the three spans, once, shared by every arm -----------------------------
# Object-axis arms do not change the blob, so this is paid once per script run and not per arm.
for n in "$N_LO" "$N_DENSE" "$N_MOE"; do
  if [ -f "$WORK/n$n/model.pkt" ] && [ "${REEMIT:-0}" != 1 ]; then
    echo "  asset n=$n: reusing $WORK/n$n"; continue
  fi
  echo "  asset n=$n: emitting"
  # shellcheck disable=SC2086
  env PLOW_LAYERS="$n" $EMIT_ENV "$NIX" develop "$ROOT" -c "$PLOWC" \
      --emit devblob --hf-dir "$CKPT" --gpu MI300X --arch gfx942 \
      --num-gpus "$TP" --max-ctx "$MAXCTX" --out "$WORK/n$n" > "$WORK/emit_n$n.log" 2>&1 \
    || { echo "FAIL: emit n=$n (see $WORK/emit_n$n.log)"; tail -20 "$WORK/emit_n$n.log"; exit 1; }
  grep -a "glm52-FULL" "$WORK/emit_n$n.log" | sed 's/^/    /'
  # The tuning store is silent when it goes stale (RECIPE §5) and a stale store means the GEMM
  # tiles came from the analytical model. It cancels in a ranking; it must not be invisible.
  grep -aq "skipped as STALE" "$WORK/emit_n$n.log" \
    && echo "    NOTE: tuning store STALE for this build -> tiles are ANALYTICAL"
done

# TILE PROVENANCE MUST AGREE ACROSS THE THREE SPANS, and this is not a formality — it is the one
# way the differencing can be corrupted by something outside this script. The score is a
# DIFFERENCE of two blobs, so it is only a per-layer cost if both were compiled with the same GEMM
# tile selection. The tuning store is repopulated by a separate campaign runner
# (`scripts/rebench_tune_gemm_gfx942.sh`) and `pick_tile` silently falls back to the analytical
# model while it is stale, so two assets emitted twenty minutes apart can disagree. Observed
# exactly once, here: n=1 emitted with no provenance at all (stale store, analytical tiles) while
# n=3/n=11 came out `measured`, which put an unknown constant inside L_dense.
prov="$(for n in "$N_LO" "$N_DENSE" "$N_MOE"; do
          python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tuning'].get('tile_source','ABSENT'))" \
            "$WORK/n$n/build.json"; done | sort -u | tr '\n' ' ')"
if [ "$(echo "$prov" | wc -w)" -ne 1 ]; then
  echo "FAIL: the three spans disagree on GEMM TILE PROVENANCE ($prov)."
  echo "      Their difference is then a tile-selection change, not a per-layer cost."
  echo "      Re-emit all three in one pass: REEMIT=1 $0 $MATRIX"
  exit 1
fi
echo "  tile provenance (all three spans): $prov"

# ---- phase 2: GATE 1, the GPU lock --------------------------------------------------------
HAVE_LOCK=0
# The trap must `exit`: releasing and then continuing UNLOCKED makes the later EXIT delete a lock
# that by then belongs to someone else.
release() { [ "$HAVE_LOCK" = 1 ] && rm -rf "$LOCK"; return 0; }
trap 'release; exit 143' INT TERM
trap 'release' EXIT
echo "waiting for the GPU lock ($LOCK) — another agent holding it for a long time is NORMAL"
for i in $(seq 1 "${LOCK_TRIES:-720}"); do
  mkdir "$LOCK" 2>/dev/null && { HAVE_LOCK=1; break; }
  sleep 5
done
[ "$HAVE_LOCK" = 1 ] || { echo "FAIL: could not take the GPU lock"; exit 1; }
echo "$$ glm52-block-sweep" > "$LOCK/owner" 2>/dev/null
# `pgrep -x` is comm-exact and misses a renamed binary; `pgrep -f "plowrt serve"` self-matches.
if pgrep '^plowrt' >/dev/null 2>&1; then
  echo "FAIL: a plowrt is already running:"; pgrep -a '^plowrt'; exit 1
fi

# ---- phase 3: measure, SERIAL (one box), one nix shell for the whole sweep -----------------
# One `nix develop` for the whole run: its startup is ~2 s and there are 3 x arms invocations.
# LD_LIBRARY_PATH must be exported INSIDE the shell — set outside, it does not survive
# `nix develop`, and plowrt then cannot dlopen libhsa (RECIPE gate 2).
{
  echo "set -u"
  echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:$ROCM_LIB\""
  echo "export RUST_LOG=info"
  echo "run_one() {  # <arm> <objdir> <n> <env...>"
  echo "  arm=\$1; obj=\$2; n=\$3; shift 3"
  echo "  log=\"$WORK/\$arm.n\$n.log\""
  echo "  if env PLOW_MLA_PF_V2=1 \"\$@\" \"$PLOWRT\" amd-probe \\"
  echo "        --blob \"$WORK/n\$n/model.pkt\" --hsaco \"\$obj\" --tp $TP --steps 0 \\"
  echo "        --prefill-sweep \"$CTXS\" --prefill-reps $REPS > \"\$log\" 2>&1; then"
  echo "    echo \"  ran \$arm n=\$n: \$(grep -ac PFSWEEP \"\$log\") point(s)\""
  echo "  else"
  echo "    echo \"  RUNFAIL \$arm n=\$n (see \$log)\"; tail -5 \"\$log\""
  echo "  fi"
  echo "}"
  for i in "${!names[@]}"; do
    for n in "$N_LO" "$N_DENSE" "$N_MOE"; do
      printf 'run_one %q %q %q %s\n' "${names[$i]}" "${objs[$i]}" "$n" "${envs[$i]}"
    done
  done
} > "$WORK/.sweep.sh"
"$NIX" develop "$ROOT" -c bash "$WORK/.sweep.sh"

# ---- phase 4: GATE 2 + the object manifest + parse ----------------------------------------
printf 'arm\tn_layers\tctx\tmedian_ms\tspread_pct\n' > "$WORK/raw.tsv"
# `tracing_subscriber` writes ANSI even to a file, so "arch=gfx942" is never literal in the log.
# A gate that greps the raw bytes for it fails OPEN on the day someone removes the colours and
# CLOSED every other day; strip the escapes first and grep the text.
plain() { sed -e 's/\x1b\[[0-9;]*m//g' "$1"; }
for i in "${!names[@]}"; do
  arm="${names[$i]}"
  for n in "$N_LO" "$N_DENSE" "$N_MOE"; do
    log="$WORK/$arm.n$n.log"
    [ -f "$log" ] || { echo "FAIL: no log for $arm n=$n"; exit 1; }
    txt="$(plain "$log")"
    case "$txt" in *"CPU reference backend"*)
      echo "FAIL: $arm n=$n selected the CPU REFERENCE BACKEND — every number would be fiction"; exit 1;; esac
    echo "$txt" | grep -q "AMD engine ready" || { echo "FAIL: $arm n=$n: engine never came up"; tail -5 "$log"; exit 1; }
    echo "$txt" | grep "AMD engine ready" | grep -q "arch=gfx942" || {
      echo "FAIL: $arm n=$n: the engine did not come up on gfx942"; exit 1; }
    echo "$txt" | grep "PFSWEEP" \
      | sed -E 's/.*PFSWEEP T=([0-9]+) median_ms=([0-9.]+) spread_pct=([0-9.]+).*/\1\t\2\t\3/' \
      | awk -v a="$arm" -v n="$n" -F'\t' '{printf "%s\t%s\t%s\t%s\t%s\n", a, n, $1, $2, $3}' >> "$WORK/raw.tsv"
  done
  # THE OBJECT MANIFEST. `Variant::detect` picks the interpreter by scanning opcodes, and a knob
  # compiled into an object the run never opens reports a confident null. GLM-5.2's fp8 is
  # BLOCK-scaled and its MLA is bf16, so `detect` reports Bf16 and the _fp8_ objects are NOT the
  # ones that run — which is exactly the trap that cost this campaign a whole round.
  echo "  [$arm] objects opened: $(plain "$WORK/$arm.n$N_MOE.log" \
       | grep -o 'object=[a-z0-9_.]*\.elf' | sort -u | tr '\n' ' ')"
done

# ---- phase 5: score -----------------------------------------------------------------------
OUT="$OUT" RAW="$WORK/raw.tsv" N_LO="$N_LO" N_DENSE="$N_DENSE" N_MOE="$N_MOE" \
MODEL_DENSE="$MODEL_DENSE" MODEL_MOE="$MODEL_MOE" python3 - <<'PY'
import os, collections
raw, out = os.environ["RAW"], os.environ["OUT"]
nlo, nde, nmo = int(os.environ["N_LO"]), int(os.environ["N_DENSE"]), int(os.environ["N_MOE"])
MD, MM = int(os.environ["MODEL_DENSE"]), int(os.environ["MODEL_MOE"])
t = collections.defaultdict(dict)   # (arm, ctx) -> {n: ms}
spread = {}
arms, ctxs = [], []
for line in open(raw).read().splitlines()[1:]:
    a, n, c, ms, sp = line.split("\t")
    t[(a, int(c))][int(n)] = float(ms)
    spread[(a, int(c), int(n))] = float(sp)
    if a not in arms: arms.append(a)
    if int(c) not in ctxs: ctxs.append(int(c))
ctxs.sort()

rows = []
for a in arms:
    for c in ctxs:
        d = t[(a, c)]
        if not all(k in d for k in (nlo, nde, nmo)):
            print(f"  {a} ctx={c}: INCOMPLETE, skipped"); continue
        ldense = (d[nde] - d[nlo]) / (nde - nlo)
        lmoe   = (d[nmo] - d[nde]) / (nmo - nde)
        score  = MD * ldense + MM * lmoe
        over   = d[nlo] - ldense
        rows.append(dict(arm=a, ctx=c, l_dense=ldense, l_moe=lmoe, score=score, over=over,
                         t_lo=d[nlo], t_de=d[nde], t_mo=d[nmo],
                         sp=max(spread[(a, c, n)] for n in (nlo, nde, nmo))))

with open(out, "w") as f:
    f.write("arm\tctx\tL_dense_ms\tL_moe_ms\tscore_ms\tO_fixed_ms\t"
            f"T_n{nlo}_ms\tT_n{nde}_ms\tT_n{nmo}_ms\tmax_spread_pct\n")
    for r in rows:
        f.write(f"{r['arm']}\t{r['ctx']}\t{r['l_dense']:.4f}\t{r['l_moe']:.4f}\t{r['score']:.2f}\t"
                f"{r['over']:.3f}\t{r['t_lo']:.3f}\t{r['t_de']:.3f}\t{r['t_mo']:.3f}\t{r['sp']:.2f}\n")

print()
print(f"=== marginal per-layer cost and the kind-weighted score "
      f"({MD}*L_dense + {MM}*L_moe) ===")
print(f"{'arm':<14}{'ctx':>6}{'L_dense ms':>12}{'L_moe ms':>11}{'score ms':>11}"
      f"{'O fixed ms':>12}{'spread%':>9}")
for r in rows:
    print(f"{r['arm']:<14}{r['ctx']:>6}{r['l_dense']:>12.4f}{r['l_moe']:>11.4f}"
          f"{r['score']:>11.1f}{r['over']:>12.3f}{r['sp']:>9.2f}")

base = arms[0]
if len(arms) > 1:
    print()
    print(f"=== ranking vs '{base}' (negative = faster). THIS IS A COMPARATOR, NOT A TTFT. ===")
    print(f"{'arm':<14}{'ctx':>6}{'d_score%':>10}{'d_L_moe%':>10}{'d_L_dense%':>12}")
    for c in ctxs:
        b = next((r for r in rows if r['arm'] == base and r['ctx'] == c), None)
        if not b: continue
        for a in arms[1:]:
            r = next((x for x in rows if x['arm'] == a and x['ctx'] == c), None)
            if not r: continue
            print(f"{a:<14}{c:>6}{100*(r['score']/b['score']-1):>10.2f}"
                  f"{100*(r['l_moe']/b['l_moe']-1):>10.2f}"
                  f"{100*(r['l_dense']/b['l_dense']-1):>12.2f}")
print(f"\nwrote {out}")
print("\nThe score omits every term that does not scale with the layer chain (embed, lm_head,")
print("sampler, host chunk plan, one-time buffer touches) — that is `O`, printed above. It is")
print("constant across knobs, so it cancels in a RANKING and ONLY the ranking is claimed.")
PY
