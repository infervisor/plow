#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-walk-b16-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"/{b8ctl,b16ctl,b16walk,obj8,obj16,objwalk,checkpoint}
for arm in b8ctl b16ctl b16walk; do
  touch "$TMP/$arm"/{model.pkt,build.json,weights.json}
done
touch "$TMP"/obj8/interp_decode.elf "$TMP"/obj16/interp_decode.elf "$TMP"/objwalk/interp_decode.elf

cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'lease\n' >>"${WALK_SELFTEST_LEASE_COUNT:?}"
shift 3
exec env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 "$@"
SH
cat >"$TMP/nix" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while [[ "$1" != --command ]]; do shift; done
shift
exec "$@"
SH
cat >"$TMP/plowrt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "${ROCR_VISIBLE_DEVICES:-}" == 0 && "${HIP_VISIBLE_DEVICES:-}" == 0 ]]
checkpoint= objects= assets= prompts= output_len=
while (($#)); do
  case "$1" in
    --rt-checkpoint) checkpoint=$2; shift 2 ;;
    --rt-hsaco) objects=$2; shift 2 ;;
    --assets) assets=$2; shift 2 ;;
    --prompt-rows) prompts=$2; shift 2 ;;
    --output-len) output_len=$2; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$checkpoint" "$objects" "$assets" "$prompts" "$output_len" <<'PY'
import json, os, sys
checkpoint, objects, assets, prompts_path, output_len = sys.argv[1:]
output_len = int(output_len)
with open(prompts_path) as source:
    prompts = [[int(token) for token in row.split(",")] for row in source.read().splitlines()]
batch = len(prompts)
outputs = [[7 + (slot % 2), *([11 + (slot % 2)] * (output_len - 1))] for slot in range(batch)]
selections = [{"occupied_rows": 1, "bucket": batch, "elapsed_ns": 9_000_000, "steps": 1}]
selections += [{"occupied_rows": batch, "bucket": batch, "elapsed_ns": 2_000_000, "steps": 1}
               for _ in range(4)]
selections += [{"occupied_rows": batch, "bucket": batch, "elapsed_ns": 1_000_000, "steps": 1}
               for _ in range(65)]
if os.environ.get("WALK_SELFTEST_PARTIAL"):
    selections[-1]["occupied_rows"] = batch - 1
print(json.dumps({
    "schema": "plowrt.bench.v1", "vendor": "Some(Amd)", "num_gpus": 1, "parallel": "tp",
    "asset_dir": assets, "requests": batch, "completed": batch, "failed": 0,
    "warmup_requests": 0, "prompt_tokens": sum(map(len, prompts)),
    "output_tokens": batch * output_len, "scheduler": {"rejected": 0, "admit_shed": 0},
    "engine": {"batch_capacity": batch},
    "token_audit": {"prompt_token_ids": prompts, "output_token_ids": outputs},
    "diagnostics": {"supported": True, "complete": True, "overflowed": False,
                    "decode_selections": selections},
    "artifacts": {
        "packet": {"path": os.path.join(assets, "model.pkt"), "checksum": "packet"},
        "build_manifest": {"path": os.path.join(assets, "build.json"), "checksum": "build"},
        "weights_manifest": {"path": os.path.join(assets, "weights.json"), "checksum": "weights"},
        "checkpoint": {"path": checkpoint, "layout_checksum": "checkpoint"},
        "object_inventory": [{"path": os.path.join(objects, "interp_decode.elf"), "checksum": "object"}],
    },
}))
PY
SH
chmod +x "$TMP/gpulease" "$TMP/nix" "$TMP/plowrt"

run_gate() {
  AB="$TMP" GEMMA31B="$TMP/checkpoint" PLOWRT="$TMP/plowrt" \
    PLOW_NIX_BIN="$TMP/nix" PLOW_GPULEASE_BIN="$TMP/gpulease" \
    WALK_B8_HSACO="$TMP/obj8" WALK_B16_HSACO="$TMP/obj16" \
    WALK_B16_WALK_HSACO="$TMP/objwalk" WALK_SELFTEST_LEASE_COUNT="$TMP/lease-count" \
    WALK_SELFTEST_PARTIAL="${WALK_SELFTEST_PARTIAL:-}" \
    bash "$ROOT/scripts/walk_b16_ab.sh" run
}

: >"$TMP/lease-count"
run_gate >"$TMP/pass.log"
[[ "$(wc -l <"$TMP/lease-count")" -eq 1 ]]
grep -q '65 dispatches x batch 8 after 4 warmups' "$TMP/pass.log"
grep -q 'aggregate 16000.0 tok/s' "$TMP/pass.log"

: >"$TMP/lease-count"
if WALK_SELFTEST_PARTIAL=1 run_gate >"$TMP/fail.log" 2>&1; then
  echo "FAIL: partial timed dispatch passed" >&2
  exit 1
fi
[[ "$(wc -l <"$TMP/lease-count")" -eq 1 ]]
grep -q 'timed window contains a partial' "$TMP/fail.log"
echo "PASS: walk B16 uses one lease and rejects partial timed dispatches"
