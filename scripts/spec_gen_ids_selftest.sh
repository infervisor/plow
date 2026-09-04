#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-spec-gen-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

cat >"$TMP/nix" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo "nix-shell banner"
while [[ "$1" != "--command" ]]; do shift; done
shift
exec "$@"
SH
cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while (($#)); do
  if [[ "$1" == "-c" ]]; then
    shift
    exec bash -c "$1"
  fi
  shift
done
exit 2
SH
cat >"$TMP/plowrt" <<'SH'
#!/usr/bin/env bash
if [[ " $* " == *" --token-audit "* ]]; then
  printf '%s\n' '{"schema":"plowrt.bench.v1","requests":4,"completed":4,"failed":0,"token_audit":{"prompt_token_ids":[[2,106,1645],[2,106,1645],[2,106,1645],[2,106,1645]],"output_token_ids":[[7,11,13,17],[7,11,13,17],[7,11,13,17],[7,11,13,17]]},"engine":{"batch_capacity":4},"diagnostics":{"supported":true,"complete":true,"overflowed":false,"decode_selections":[{"occupied_rows":4,"bucket":4}]}}'
else
  printf '%s\n' '{"parity":{"output_token_ids":[[7,11,13,17]]}}'
fi
SH
chmod +x "$TMP/nix" "$TMP/gpulease" "$TMP/plowrt"

output="$({
  PLOW_NIX_BIN="$TMP/nix" \
  PLOW_GPULEASE_BIN="$TMP/gpulease" \
  PLOWRT_BIN="$TMP/plowrt" \
  SPEC_GEN_REPORT="$TMP/report.json" \
    "$ROOT/scripts/spec_gen_ids.sh" "$TMP/assets" 4
} 2>&1)"

[[ "$output" == *"nix-shell banner"* ]]
[[ "$output" == *"[7, 11, 13, 17]"* ]]
python3 - "$TMP/report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    assert json.load(source)["parity"]["output_token_ids"] == [[7, 11, 13, 17]]
PY

gate_output="$({
  PLOW_NIX_BIN="$TMP/nix" \
  PLOWRT_BIN="$TMP/plowrt" \
  GATE_QUICK_OUT="$TMP/gate-report.json" \
    bash "$ROOT/scripts/gate_quick.sh"
} 2>&1)"
[[ "$gate_output" == *"nix-shell banner"* ]]
[[ "$gate_output" == *"PASS: 4 production-mux slots"* ]]
python3 - "$TMP/gate-report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    assert json.load(source)["completed"] == 4
PY
echo "PASS: nix stdout stays outside both JSON reports"
