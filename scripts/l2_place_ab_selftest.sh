#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-l2-place-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"/{off,on,objs_off,objs_on,checkpoint}
touch "$TMP"/{off,on}/model.pkt "$TMP"/objs_{off,on}/interp_decode.elf

cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'acquired\n' >>"${L2_SELFTEST_LEASE_COUNT:?}"
shift 3
exec env ROCR_VISIBLE_DEVICES=mock HIP_VISIBLE_DEVICES=mock "$@"
SH
cat >"$TMP/plowrt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "${ROCR_VISIBLE_DEVICES:-}" == mock && "${HIP_VISIBLE_DEVICES:-}" == mock ]]
assets= prompt= steps= warmups=
while (($#)); do
  case "$1" in
    --assets) assets=$2; shift 2 ;;
    --prompt-ids) prompt=$2; shift 2 ;;
    --output-len) steps=$2; shift 2 ;;
    --warmup-requests) warmups=$2; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$assets" "$prompt" "$steps" "$warmups" <<'PY'
import json, os, sys
assets, prompt, steps, warmups = sys.argv[1:]
prompt = [int(token) for token in prompt.split(",")]
steps, warmups = int(steps), int(warmups)
row = list(range(7, 7 + steps))
if os.environ.get("L2_SELFTEST_MISMATCH") and os.environ["PLOW_L2_PLACE_DISPATCH"] == "1":
    row[-1] += 1
print(json.dumps({
    "schema": "plowrt.bench.v1", "vendor": "Some(Amd)",
    "requests": 1, "completed": 1, "failed": 0, "warmup_requests": warmups,
    "prompt_tokens": len(prompt), "output_tokens": steps,
    "scheduler": {"rejected": 0, "admit_shed": 0},
    "tpot_ms": {"p50": 1.25},
    "artifacts": {
        "packet": {"path": os.path.join(assets, "model.pkt")},
        "checkpoint": {"path": os.environ["PLOW_CHECKPOINT"]},
        "object_inventory": [{"path": os.path.join(os.environ["PLOW_HSACO"], "interp_decode.elf")}],
    },
    "runtime": {"environment": [
        ["PLOW_HSACO", os.environ["PLOW_HSACO"]],
        ["PLOW_CHECKPOINT", os.environ["PLOW_CHECKPOINT"]],
        ["PLOW_L2_PLACE_DISPATCH", os.environ["PLOW_L2_PLACE_DISPATCH"]],
    ]},
    "token_audit": {"prompt_token_ids": [prompt], "output_token_ids": [row]},
}))
PY
SH
chmod +x "$TMP/gpulease" "$TMP/plowrt"

LEASE_COUNT="$TMP/lease-count"
: >"$LEASE_COUNT"
PLOW_AB_DIR="$TMP" PLOW_CKPT="$TMP/checkpoint" PLOWRT_BIN="$TMP/plowrt" \
  PLOW_GPULEASE_BIN="$TMP/gpulease" L2_SELFTEST_LEASE_COUNT="$LEASE_COUNT" \
  PROMPT=2,3,5 STEPS=4 FOLDS=1 bash "$ROOT/scripts/l2_place_ab.sh" \
  >"$TMP/pass.log"
grep -q "TOKEN-IDENTICAL" "$TMP/pass.log"
[[ "$(wc -l <"$LEASE_COUNT")" -eq 1 ]]

: >"$LEASE_COUNT"
if L2_SELFTEST_MISMATCH=1 PLOW_AB_DIR="$TMP" PLOW_CKPT="$TMP/checkpoint" \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  L2_SELFTEST_LEASE_COUNT="$LEASE_COUNT" PROMPT=2,3,5 STEPS=4 FOLDS=1 \
    bash "$ROOT/scripts/l2_place_ab.sh" >"$TMP/fail.log" 2>&1; then
  echo "FAIL: mismatched placement streams passed" >&2
  exit 1
fi
grep -q "MISMATCH" "$TMP/fail.log"
[[ "$(wc -l <"$LEASE_COUNT")" -eq 1 ]]
echo "PASS: L2 production A/B uses one lease and rejects a token mismatch"
