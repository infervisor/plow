#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-l1-tokens-selftest.XXXXXX)"
trap 'rm -rf "$TMP" /tmp/l1-tok-{base,d1,d8}.txt' EXIT

mkdir -p "$TMP"/{base,d1,d8,base-hsaco,hsaco,checkpoint}
touch "$TMP"/{base,d1,d8}/model.pkt "$TMP/base-hsaco/interp_decode.elf" \
  "$TMP/hsaco/interp_decode.elf"

cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
shift 3
exec "$@"
SH
cat >"$TMP/plowrt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
assets= prompt= steps=
while (($#)); do
  case "$1" in
    --assets) assets=$2; shift 2 ;;
    --prompt-ids) prompt=$2; shift 2 ;;
    --output-len) steps=$2; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$assets" "$PLOW_HSACO" "$PLOW_CHECKPOINT" "$prompt" "$steps" <<'PY'
import json, os, sys
assets, hsaco, checkpoint, prompt, steps = sys.argv[1:]
prompt = [int(token) for token in prompt.split(",")]
output = [7, 11, 13, 17][:int(steps)]
if os.environ.get("L1_SELFTEST_INCOMPLETE"):
    output = output[:-1]
print(json.dumps({
    "schema": "plowrt.bench.v1", "vendor": "Some(Amd)",
    "asset_dir": assets, "requests": 1, "completed": 1, "failed": 0,
    "warmup_requests": 0,
    "artifacts": {
        "packet": {"path": os.path.join(assets, "model.pkt")},
        "checkpoint": {"path": checkpoint},
        "object_inventory": [{"path": os.path.join(hsaco, "interp_decode.elf")}],
    },
    "token_audit": {"prompt_token_ids": [prompt], "output_token_ids": [output]},
}))
PY
SH
chmod +x "$TMP/gpulease" "$TMP/plowrt"

output="$(
  CK="$TMP/checkpoint" STEPS=4 PROMPT=2,106,1645 \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  L1_BASE_ASSETS="$TMP/base" L1_D1_ASSETS="$TMP/d1" L1_D8_ASSETS="$TMP/d8" \
  L1_BASE_HSACO="$TMP/base-hsaco" L1_HSACO="$TMP/hsaco" \
    bash "$ROOT/scripts/l1_tokens.sh"
)"
[[ "$output" == *"d1 == base  TOKEN-IDENTICAL"* ]]
[[ "$output" == *"d8 == base  TOKEN-IDENTICAL"* ]]

if CK="$TMP/checkpoint" STEPS=4 PROMPT=2,106,1645 L1_SELFTEST_INCOMPLETE=1 \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  L1_BASE_ASSETS="$TMP/base" L1_D1_ASSETS="$TMP/d1" L1_D8_ASSETS="$TMP/d8" \
  L1_BASE_HSACO="$TMP/base-hsaco" L1_HSACO="$TMP/hsaco" \
    bash "$ROOT/scripts/l1_tokens.sh" >"$TMP/incomplete.log" 2>&1; then
  echo "FAIL: incomplete token audit passed" >&2
  exit 1
fi
grep -q "token audit returned an incomplete stream" "$TMP/incomplete.log"
echo "PASS: L1 token gate uses complete artifact-bound production bench reports"
