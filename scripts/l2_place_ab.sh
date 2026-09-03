#!/usr/bin/env bash
# Interleaved production-mux A/B for L2-domain placement on Gemma-4-31B decode, gfx950.
#
# Arm A: unplaced packet + objects built without PLOW_L2_PLACE_DISPATCH.
# Arm B: placed packet + objects built with it and the matching runtime guard.
# Packet, object, checkpoint, prompt, and complete output identities are checked
# before timing or token equality is accepted.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${PLOW_AB_DIR:-/tmp/l2place}"
CKPT="${PLOW_CKPT:-$(readlink -f /home/lava/plow/build-amd/g31b-bf16/checkpoint)}"
RT="${PLOWRT_BIN:-$REPO/target/release/plowrt}"
LEASE="${PLOW_GPULEASE_BIN:-$REPO/perf-data/tools/gpulease}"
STEPS="${STEPS:-64}"
FOLDS="${FOLDS:-3}"
WARMUPS="${WARMUPS:-1}"
PROMPT="${PROMPT:-2,106,1645,108,7154,1701,532,573,6996,529,8043,236881,107,108,106,2516,108}"

if [[ -z "${PLOW_L2_PLACE_LEASED:-}" ]]; then
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec env PLOW_L2_PLACE_LEASED=1 "$LEASE" -n 1 l2-place-ab "$0" "$@"
fi
mkdir -p "$D"

run() { # <tag> <packet> <objects> <placement-guard>
  local tag=$1 packet=$2 objects=$3 placement=$4 assets report log ids
  assets="$(dirname "$packet")"
  report="$D/raw.$tag.json"
  log="$D/raw.$tag.log"
  ids="$D/ids.$tag.txt"
  env PLOW_L2_PLACE_DISPATCH="$placement" PLOW_HSACO="$objects" PLOW_CHECKPOINT="$CKPT" \
    "$RT" bench --assets "$assets" --prompt-ids "$PROMPT" \
    --concurrency 1 --requests 1 --warmup-requests "$WARMUPS" \
    --output-len "$STEPS" --token-audit >"$report" 2>"$log" || {
      cat "$log" >&2
      return 1
    }
  python3 - "$report" "$ids" "$tag" "$packet" "$objects" "$CKPT" \
    "$PROMPT" "$STEPS" "$WARMUPS" "$placement" <<'PY'
import json, os, sys

(report_path, ids_path, tag, packet, objects, checkpoint, prompt,
 steps, warmups, placement) = sys.argv[1:]
with open(report_path) as source:
    report = json.load(source)

def need(ok, message):
    if not ok:
        raise SystemExit(f"{tag}: {message}")

prompt = [int(token.strip()) for token in prompt.split(",")]
steps, warmups = int(steps), int(warmups)
audit = report.get("token_audit") or {}
rows = audit.get("output_token_ids")
need(report.get("schema") == "plowrt.bench.v1", "unexpected bench schema")
need(report.get("vendor") == "Some(Amd)", "production bench did not use AMD")
need((report.get("requests"), report.get("completed"), report.get("failed")) == (1, 1, 0),
     "measured request did not complete exactly once")
need(report.get("warmup_requests") == warmups, "warmup count changed")
need(report.get("prompt_tokens") == len(prompt), "prompt token count changed")
need(report.get("output_tokens") == steps, "output token count changed")
need(audit.get("prompt_token_ids") == [prompt], "token audit prompt changed")
need(isinstance(rows, list) and len(rows) == 1 and len(rows[0]) == steps,
     "token audit returned an incomplete stream")
need(any(rows[0]), "all-zero output makes token identity vacuous")
schedule = report.get("scheduler") or {}
need(schedule.get("rejected") == 0 and schedule.get("admit_shed") == 0,
     "production scheduler rejected or shed work")

artifacts = report.get("artifacts") or {}
observed_packet = (artifacts.get("packet") or {}).get("path")
observed_checkpoint = (artifacts.get("checkpoint") or {}).get("path")
inventory = artifacts.get("object_inventory") or []
need(os.path.realpath(observed_packet or "") == os.path.realpath(packet),
     "bench measured a different packet")
need(os.path.realpath(observed_checkpoint or "") == os.path.realpath(checkpoint),
     "bench bound a different checkpoint")
object_root = os.path.realpath(objects)
need(inventory, "bench reported no code objects")
need(any(os.path.commonpath([os.path.realpath(item.get("path", "")), object_root]) == object_root
         for item in inventory), "bench did not inventory the selected object directory")
environment = dict((report.get("runtime") or {}).get("environment") or [])
need(os.path.realpath(environment.get("PLOW_HSACO", "")) == object_root,
     "runtime object override changed")
need(os.path.realpath(environment.get("PLOW_CHECKPOINT", "")) == os.path.realpath(checkpoint),
     "runtime checkpoint override changed")
need(environment.get("PLOW_L2_PLACE_DISPATCH") == placement,
     "runtime placement guard changed")

with open(ids_path, "w") as output:
    output.write(",".join(map(str, rows[0])) + "\n")
print(f"[{tag}] p50 TPOT {report['tpot_ms']['p50']:.3f} ms; report {report_path}")
PY
}

for fold in $(seq 1 "$FOLDS"); do
  echo "########## fold $fold ##########"
  run "A$fold" "$D/off/model.pkt" "$D/objs_off" 0
  run "B$fold" "$D/on/model.pkt" "$D/objs_on" 1
  if cmp -s "$D/ids.A$fold.txt" "$D/ids.B$fold.txt"; then
    echo "A$fold == B$fold  TOKEN-IDENTICAL"
  else
    echo "A$fold != B$fold  MISMATCH" >&2
    exit 1
  fi
done
