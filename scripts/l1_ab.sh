#!/usr/bin/env bash
# L1 synthetic A/B: flash_merge 32 -> 256 workgroups, interleaved.
#
# The two arms differ ONLY in the BLOB (dsplit=1 vs dsplit=8); both load the SAME
# code objects, because `d_flash_merge` derives its D-split from the workgroup count
# it is handed. That is the §7a methodology: same binary, one variable.
#
# Interleaved A,B,A,B,... so a drift in clocks or a neighbour's load hits both arms.
# rc=76 from gpulease is contention -> the caller must re-run, never report it.
# This is an unbound packet/kernel probe at an artificial context. Its timings
# are valid only for the internal A/B and are not served-model performance.
#
#   l1_ab.sh <reps> [ctx] [steps]
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPS="${1:-6}"; CTX="${2:-1024}"; STEPS="${3:-64}"
HSACO=/home/lava/plow/build-amd/l1-gfx950
OUT="${OUT:-/tmp/l1-ab.txt}"
: > "$OUT"

run() { # <arm>
  local arm=$1 dir hs log rc
  # The `base` arm is the REGRESSION control: pre-change objects + pre-change blob.
  # Every other arm is post-change objects + that arm's blob, so base-vs-d1 prices
  # the kernel edit itself and d1-vs-dN prices the widening.
  if [ "$arm" = base ]; then
    dir=/home/lava/plow/build-amd/l1-basepkt; hs=/home/lava/plow/build-amd/l1-base
  else
    dir=/home/lava/plow/build-amd/l1-$arm;    hs=$HSACO
  fi
  log=$(mktemp)
  set +e
  "$W/perf-data/tools/gpulease" -n 1 "l1-$arm" \
      "$W/target/release/plowrt" amd-probe \
      --blob "$dir/model.pkt" --hsaco "$hs" \
      --ctx "$CTX" --steps "$STEPS" >"$log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 76 ]; then echo "$arm CONTENDED rc=76" | tee -a "$OUT"; rm -f "$log"; return 0; fi
  if grep -q "CONTENDED" "$log"; then echo "$arm CONTENDED audit" | tee -a "$OUT"; rm -f "$log"; return 0; fi
  local ms
  ms=$(grep -oP 'decode steps at ctx=[0-9]+: \K[0-9.]+' "$log" | head -1)
  [ -z "$ms" ] && { echo "$arm PARSE-FAIL rc=$rc"; tail -5 "$log"; rm -f "$log"; return 1; }
  echo "$arm $ms" | tee -a "$OUT"
  rm -f "$log"
}

ARMS="${ARMS:-d1 d8}"
for i in $(seq 1 "$REPS"); do
  for a in $ARMS; do run "$a"; done
done

python3 - "$OUT" <<'PY'
import sys, statistics as st
rows = {}
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) == 2:
        try: rows.setdefault(f[0], []).append(float(f[1]))
        except ValueError: pass
for k in sorted(rows):
    v = sorted(rows[k])
    print(f"{k}: n={len(v)} median={st.median(v):.3f} min={v[0]:.3f} max={v[-1]:.3f} "
          f"sd={st.pstdev(v):.3f}  {['%.3f'%x for x in v]}")
ref = 'base' if 'base' in rows else 'd1'
if ref in rows:
    b = st.median(rows[ref])
    for k in sorted(rows):
        if k != ref:
            print(f"delta(median {k} - {ref}) = {st.median(rows[k]) - b:+.3f} ms/token")
PY
