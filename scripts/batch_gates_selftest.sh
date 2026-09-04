#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

make_assets() { # <dir> <capacity>
  mkdir -p "$1" "$1-hsaco" "$TMP/checkpoint"
  : >"$1/model.pkt"
  : >"$1-hsaco/interp_decode.elf"
  printf '{"shapes":{"decode_batch":%s}}\n' "$2" >"$1/build.json"
}
for width in 1 4 8; do make_assets "$TMP/g-db$width" "$width"; done
make_assets "$TMP/k3-b4" 4
make_assets "$TMP/k3-b8" 8

cat >"$TMP/mock-plowrt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
checkpoint= hsaco= assets= rows= requests= output=
while (($#)); do
  case "$1" in
    --rt-checkpoint) checkpoint=$2; shift 2 ;;
    --rt-hsaco) hsaco=$2; shift 2 ;;
    --assets) assets=$2; shift 2 ;;
    --prompt-rows) rows=$2; shift 2 ;;
    --requests) requests=$2; shift 2 ;;
    --output-len) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$assets" "$hsaco" "$checkpoint" "$rows" "$requests" "$output" <<'PY'
import json, os, sys
assets, hsaco, checkpoint, rows_path, requests, output = sys.argv[1:]
requests, output = int(requests), int(output)
with open(rows_path) as source:
    prompts = [[int(x.strip()) for x in line.split(",")] for line in source.read().splitlines()]
outputs = [[sum(prompt) % 997 + step + 1 for step in range(output)] for prompt in prompts]
mode = os.environ.get("MOCK_MODE", "valid")
completed = requests
if mode == "incomplete":
    completed -= 1
    outputs.pop()
if mode == "misordered":
    prompts.reverse()
if mode == "ragged-mismatch" and os.path.basename(assets) == "k3-b8":
    outputs[1][0] += 1
lengths = [len(row) for row in prompts]
report = {
    "schema": "plowrt.bench.v1", "vendor": "Some(Amd)", "num_gpus": 8,
    "asset_dir": assets, "requests": requests, "completed": completed,
    "failed": requests - completed, "warmup_requests": 0,
    "input": {"mode": "token_rows", "row_count": requests,
              "min_tokens_per_request": min(lengths), "max_tokens_per_request": max(lengths)},
    "engine": {"batch_capacity": requests},
    "diagnostics": {"supported": True, "complete": True, "overflowed": False,
                    "decode_selections": [{"occupied_rows": requests, "bucket": requests}]},
    "artifacts": {"packet": {"path": os.path.join(assets, "model.pkt"), "checksum": "mock:packet"},
                  "checkpoint": {"path": checkpoint, "layout_checksum": "mock:checkpoint"},
                  "object_inventory": [{"path": os.path.join(hsaco, "interp_decode.elf"),
                                        "checksum": "mock:object"}]},
    "token_audit": {"prompt_token_ids": prompts, "output_token_ids": outputs},
}
print(json.dumps(report))
PY
SH

cat >"$TMP/mock-nix" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while [[ "$1" != --command ]]; do shift; done
shift
exec "$@"
SH

cat >"$TMP/mock-lease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -n && "$2" == 8 ]] || { echo "expected TP8 lease" >&2; exit 1; }
shift 3
[[ "$1" == sg && "$2" == render && "$3" == -c ]] || exit 1
exec bash -c "$4"
SH
chmod +x "$TMP/mock-plowrt" "$TMP/mock-nix" "$TMP/mock-lease"

run_gemma() {
  env MOCK_MODE="$1" PLOW_REPO="$ROOT" PLOWRT_BIN="$TMP/mock-plowrt" \
    PLOW_NIX_BIN="$TMP/mock-nix" CKPT="$TMP/checkpoint" \
    G31B_ASSETS_PREFIX="$TMP/g-db" G31B_HSACO_PREFIX="$TMP/g-db" \
    "$ROOT/scripts/gate_batched.sh"
}
run_k3() {
  env MOCK_MODE="$1" PLOWRT_BIN="$TMP/mock-plowrt" PLOW_GPULEASE_BIN="$TMP/mock-lease" \
    PLOW_NIX_BIN="$TMP/mock-nix" \
    "$ROOT/scripts/k3_batch_gate.sh" "$TMP/k3-b4" "$TMP/k3-b4-hsaco" \
    "$TMP/checkpoint" 4 "$TMP/k3-b8" "$TMP/k3-b8-hsaco"
}

run_gemma valid | grep -q '^PASS:'
run_k3 valid | grep -q '^BATCH GATE: PASS'
if run_gemma incomplete >/dev/null 2>&1; then echo "incomplete report passed" >&2; exit 1; fi
if run_gemma misordered >/dev/null 2>&1; then echo "misordered report passed" >&2; exit 1; fi
if run_k3 ragged-mismatch >/dev/null 2>&1; then echo "ragged cross-width mismatch passed" >&2; exit 1; fi
echo "PASS: batch gate mocks reject incomplete, misordered, and ragged-mismatch results"
