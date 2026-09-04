#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-batch-ceiling-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/assets/g31b-db2" "$TMP/objects/hsaco-b2" "$TMP/checkpoint" "$TMP/out"
touch "$TMP/assets/g31b-db2"/{model.pkt,build.json,weights.json}
touch "$TMP/objects/hsaco-b2/interp_decode.elf"

cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'lease\n' >>"${BATCH_CEILING_SELFTEST_LEASE_COUNT:?}"
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
outputs = [[7, *([11] * (output_len - 1))] for _ in range(batch)]
selections = [{"occupied_rows": 1, "bucket": batch, "elapsed_ns": 9_000_000, "steps": 1},
              {"occupied_rows": batch, "bucket": batch, "elapsed_ns": 2_000_000, "steps": 1},
              {"occupied_rows": batch, "bucket": batch, "elapsed_ns": 1_000_000, "steps": 1},
              {"occupied_rows": batch, "bucket": batch, "elapsed_ns": 1_000_000, "steps": 1}]
checksum = "" if os.environ.get("BATCH_CEILING_SELFTEST_UNHASHED") else "packet"
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
        "packet": {"path": os.path.join(assets, "model.pkt"), "checksum": checksum},
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
  PLOW_REPO="$ROOT" PLOWRT="$TMP/plowrt" PLOW_NIX_BIN="$TMP/nix" \
    PLOW_GPULEASE_BIN="$TMP/gpulease" PLOW_BATCH_CEILING_CKPT="$TMP/checkpoint" \
    PLOW_BATCH_CEILING_ASSET_ROOT="$TMP/assets" PLOW_BATCH_CEILING_OBJECT_ROOT="$TMP/objects" \
    PLOW_BATCH_CEILING_OUT="$TMP/out" BATCH_CEILING_SELFTEST_LEASE_COUNT="$TMP/lease-count" \
    BATCH_CEILING_SELFTEST_UNHASHED="${BATCH_CEILING_SELFTEST_UNHASHED:-}" \
    BS=2 CTXS=3 STEPS=2 WARMUP_STEPS=1 bash "$ROOT/scripts/sweep_batch_ceiling.sh"
}

: >"$TMP/lease-count"
run_gate >"$TMP/pass.log"
[[ "$(wc -l <"$TMP/lease-count")" -eq 1 ]]
grep -q '2 dispatches x batch 2 at ctx=3 after 1 warmups' "$TMP/pass.log"
grep -q 'aggregate 2000.0 tok/s' "$TMP/pass.log"

: >"$TMP/lease-count"
if BATCH_CEILING_SELFTEST_UNHASHED=1 run_gate >"$TMP/fail.log" 2>&1; then
  echo "FAIL: unhashed packet passed" >&2
  exit 1
fi
[[ "$(wc -l <"$TMP/lease-count")" -eq 1 ]]
grep -q 'wrong or unhashed packet' "$TMP/fail.log"
echo "PASS: batch ceiling uses one lease and rejects incomplete artifact identity"
