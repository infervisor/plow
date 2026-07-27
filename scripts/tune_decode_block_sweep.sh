#!/usr/bin/env bash
# tune_decode_block_sweep.sh — RANK decode knobs cheaply on a single-layer block
# asset, so the expensive end-to-end confirm only has to run on the shortlist.
#
#   scripts/tune_decode_block_sweep.sh --asset /root/plow-out/blk-slide \
#     --arms "none GV_MM_MAX=16" --batch 1,4,8 --ctx 1024,32768 \
#     --out perf-data/px15-block-slide.jsonl
#
# WHY THIS IS NOT THE `gemv_lab` MISTAKE. tuning/README-decode-tuner.md §2 bans
# scoring a knob on an isolated microbench, because
# `runtime/nvidia/experiments/gemv_lab_h100.cu` — a STANDALONE kernel outside the
# interpreter — says row-blocking wins 1.4x on every decode shape and in the real
# megakernel it loses. A block asset is a different animal: `block_run` drives
# the REAL interpreter (same cubin, same opcode dispatch, same counter protocol,
# same register/occupancy footprint) over one layer's program. What made the
# standalone bench lie was that none of that was present.
#
# WHAT A BLOCK STILL CANNOT TELL YOU, and therefore what this script does NOT
# claim:
#   1. MAGNITUDE. No cross-layer L2 contention, no weight-traffic amortisation
#      over 48 layers, no host/mux overhead. Block ms does not scale to model ms.
#   2. The model's REGISTER CEILING. A block program may instantiate a different
#      arm set than the full model, so its occupancy can differ — and the
#      README's own inversions (GV_UNROLL 8 wins at occ-1 and loses at occ-2)
#      mean an optimum found at the wrong occupancy is simply wrong. This script
#      prints `-res-usage` for every object it builds so that can be checked
#      rather than assumed.
#   3. Anything numeric about the TOKENS. The isolated block has no upstream, so
#      `act.x` is never refreshed; per-step kernel time is data-independent
#      (which is what makes the ranking valid) but the outputs are meaningless.
# So: this produces an ORDERING. `scripts/tune_decode_sweep.sh` produces the
# number that gets published. A disagreement between the two is a finding, not a
# nuisance — record it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ASSET=""
# One arm per shell word; '+'-join several defines into one arm. `none` = the
# shipped recipe untouched, which is the baseline every other arm is read against.
ARMS="none"
BATCH="1,4,8"
CTX="1024,8192"
ITERS=60
WARMUP=10
PF_ITERS=3
# Rows of act.x uploaded per prefill pass. act.x is sized by the packet's largest
# PREFILL BUCKET (8192 on a 12B block), not by max_ctx, so without this a ctx
# above the bucket is rejected at upload and the block cannot be benched at all
# above 8k. See the flag's comment in examples/block_run.rs.
PF_CHUNK=8192
OUT=""
WORK=/dev/shm/plowtune/block
MEM_HEADROOM=6000     # MiB free the card must have; a block is small, but a
                      # neighbour's 27 GiB still perturbs clocks and L2.
LABEL=px15blk
DRY=0

usage() { sed -n '2,36p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --asset) ASSET="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --prefill-iters) PF_ITERS="$2"; shift 2;;
    --pf-chunk) PF_CHUNK="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage 0;;
    *) echo "unknown option $1" >&2; usage 2;;
  esac
done
[ -n "$ASSET" ] || { echo "--asset is required" >&2; exit 2; }
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
[ -f "$ASSET/block.json" ] || { echo "no block.json in $ASSET" >&2; exit 2; }

BLOCK_RUN="$ROOT/target/release/examples/block_run"
[ -x "$BLOCK_RUN" ] || { echo "no block_run at $BLOCK_RUN" >&2; exit 2; }

# WHICH libcuda. `device::cuda` tries /usr/local/cuda/compat BEFORE the distro
# path, which is right on a box whose toolkit outruns its kernel driver. On THIS
# box the relation is inverted — compat ships 580.167.08 and the kernel driver is
# 580.159.03 — so the compat load fails with CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
# before any measurement happens. Pin the distro driver, which matches exactly.
# Cannot be fixed with LD_LIBRARY_PATH: putting /usr/lib/x86_64-linux-gnu on it
# shadows nix's glibc and the binary dies in the loader instead.
if [ -z "${PLOW_LIBCUDA:-}" ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d ' ')"
  if [ -e "/usr/local/cuda/compat/libcuda.so.$drv" ]; then
    : # compat matches the driver; leave the default probe order alone
  else
    export PLOW_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
    echo "libcuda: pinned $PLOW_LIBCUDA (driver $drv; compat build differs)" >&2
  fi
fi
mkdir -p "$WORK/cubin" "$WORK/log" "$WORK/asset" "$(dirname "$OUT")"
touch "$OUT"

LAYER="$(sed -n 's/.*"layer": *\([0-9]*\).*/\1/p' "$ASSET/block.json" | head -1)"
KVH="$(sed -n 's/.*"kv_heads": *\([0-9]*\).*/\1/p' "$ASSET/block.json" | head -1)"
echo "asset  : $ASSET  (layer $LAYER, kv_heads $KVH)"
echo "arms   : $ARMS"
echo "grid   : batch[$BATCH] ctx[$CTX]  iters=$ITERS"

# Same cache discipline as tune_decode_sweep.sh: keyed by the SHA of the FULL
# define string, so an arm that names nothing builds an object byte-identical to
# the shipped one and re-running a grid re-measures without rebuilding.
build_arm() {   # $1 arm -> echoes cubin dir
  local arm="$1" defs="" one
  if [ "$arm" != "none" ]; then
    for one in ${arm//+/ }; do defs="$defs -D$one"; done
  fi
  local key; key="$(printf '%s' "$defs" | sha256sum | cut -c1-16)"
  local dir="$WORK/cubin/$key"
  if [ -f "$dir/interp_sm120.cubin" ]; then echo "$dir"; return 0; fi
  mkdir -p "$dir"; printf '%s\n' "$defs" > "$dir/defines"
  echo "  [build] $arm -> $key ($defs)" >&2
  [ "$DRY" = "1" ] && { echo "$dir"; return 0; }
  PLOW_ROOT="$ROOT" PLOW_EXTRA_DEFINES="$defs" \
    "$ROOT/scripts/build_sm120_cubin.sh" "$dir/interp_sm120.cubin" \
    >"$WORK/log/build_$key.log" 2>&1 \
    || { echo "FATAL: build failed for $arm — see $WORK/log/build_$key.log" >&2; exit 1; }
  # Registers AND occupancy, recorded per object. Trap 2 in the header: an
  # optimum found at the block's occupancy is wrong at the model's, and the only
  # way to know is to look.
  env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin cuobjdump -res-usage \
      "$dir/interp_sm120.cubin" 2>/dev/null \
    | awk '/interp_sm12011PlowProgram/{f=1}
           f && /REG:/ { for(i=1;i<=NF;i++) if($i ~ /^REG:/){sub("REG:","",$i); print $i; exit} }' \
    > "$dir/registers" || true
  echo "$dir"
}

for arm in $ARMS; do
  cdir="$(build_arm "$arm")"
  regs="$(tr -cd '0-9' < "$cdir/registers" 2>/dev/null || true)"; [ -n "$regs" ] || regs=null
  # The asset is symlinks over the SAME packet and checkpoint; only the cubin
  # changes between arms, which is the whole point — anything else varying would
  # make the ranking attribute a packet difference to a define.
  adir="$WORK/asset/$arm"; mkdir -p "$adir"
  for f in block.json model.pkt weights.json tokenizer.json checkpoint sample_sm120.cubin interp_sm120_pf.cubin; do
    [ -e "$ASSET/$f" ] && ln -sfn "$(readlink -f "$ASSET/$f")" "$adir/$f"
  done
  # GF_FULL IS A PAIR. The emitter sizes the full layers' nsplit from
  # `n_grp = heads / FA_GF_FULL`, so an arm that changes only the OBJECT define
  # leaves the packet splitting work for a different group count and the run
  # measures the disagreement instead of the knob. `--gff-packet <dir>` supplies
  # a per-arm packet emitted with the matching PLOW_FA_GF_FULL; without one, an
  # arm naming GF_FULL is refused rather than quietly mismeasured.
  case "$arm" in
    *PLOW_NV_FA_GF_FULL=*)
      gffv="${arm##*PLOW_NV_FA_GF_FULL=}"; gffv="${gffv%%+*}"
      pk="${GFF_PACKETS:-}/gff${gffv}/model.pkt"
      if [ -n "${GFF_PACKETS:-}" ] && [ -f "$pk" ]; then
        ln -sfn "$(readlink -f "$pk")" "$adir/model.pkt"
        echo "  [pair ] packet re-emitted for PLOW_FA_GF_FULL=$gffv" >&2
      else
        echo "FATAL: arm '$arm' changes the kernel's GF_FULL but no matching packet." >&2
        echo "       Set GFF_PACKETS=<dir> holding gff<N>/model.pkt per arm — see" >&2
        echo "       devgen::fa_gf_full() for why an unpaired sweep is meaningless." >&2
        exit 1
      fi;;
  esac
  ln -sfn "$cdir/interp_sm120.cubin" "$adir/interp_sm120.cubin"
  [ -f "$cdir/interp_sm120_pf.cubin" ] && ln -sfn "$cdir/interp_sm120_pf.cubin" "$adir/interp_sm120_pf.cubin"

  echo "[arm] $arm  regs=$regs"
  [ "$DRY" = "1" ] && continue

  # Wait for real headroom before taking the lease: gpulease serialises US but
  # cannot evict a holder outside our PID namespace, and a run started next to
  # 27 GiB of somebody else's weights is a contended number, not a slow one.
  waited=0
  while :; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
    [ "$((total - used))" -ge "$MEM_HEADROOM" ] && break
    [ "$waited" -ge 900 ] && { echo "  [wait] still only $((total-used)) MiB free after ${waited}s — running CAVEATED" >&2; break; }
    echo "  [wait] $((total-used)) MiB free < $MEM_HEADROOM — sleeping 30s" >&2
    sleep 30; waited=$((waited+30))
  done
  free_before="$((total - used))"

  log="$WORK/log/run_${arm}.log"
  set +e
  "$ROOT/perf-data/harness/gpulease" "${LABEL}-${arm}" \
    "$BLOCK_RUN" "$adir" bench --batch "$BATCH" --ctx "$CTX" \
      --iters "$ITERS" --warmup "$WARMUP" --prefill-iters "$PF_ITERS" \
      --pf-chunk "$PF_CHUNK" >"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "  rc=$rc DISCARDED (see $log)" >&2
    continue
  fi
  grep -E '^  B=' "$log" || true

  # block_run writes /dev/shm/block-asset/bench/sweep.json; fold each (B,ctx)
  # row into our jsonl tagged with the arm, so `jq` can rank per cell.
  python3 - "$arm" "$regs" "$LAYER" "$KVH" "$free_before" "$ASSET" "$OUT" <<'PY'
import json, sys, pathlib, datetime
arm, regs, layer, kvh, freeb, asset, out = sys.argv[1:8]
sw = json.load(open("/dev/shm/block-asset/bench/sweep.json"))["sweep"]
with open(out, "a") as f:
    for r in sw:
        f.write(json.dumps({
            "arm": arm, "layer": int(layer), "kv_heads": int(kvh),
            "asset": asset,
            "batch": r["batch"], "ctx": r["ctx"],
            "latency_us_median": r["latency_us_median"],
            "latency_us_p95": r["latency_us_p95"],
            "tok_s": r["tok_s"],
            "registers": None if regs == "null" else int(regs),
            "free_mib_before": int(freeb),
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        }, separators=(",", ":")) + "\n")
print(f"  -> appended {len(sw)} rows to {out}")
PY
done

echo
echo "done. $OUT"
echo "This is a RANKING. Confirm the shortlist end-to-end with tune_decode_sweep.sh"
echo "before anything here is published to tuning/."
