#!/usr/bin/env bash
# TOKEN IDENTITY GATE. flash_merge folds softmax partials: a wrong D-chunk bound
# corrupts tokens SILENTLY, so a perf number without this check is worthless.
#
# Three arms, production mux, same prompt, same checkpoint:
#   base : pre-change objects  + pre-change blob      (the reference)
#   d1   : post-change objects + dsplit=1 blob        (regression: the default path)
#   d8   : post-change objects + dsplit=8 blob        (the widened path)
# All three token lists must be character-for-character equal.
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK="${CK:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
STEPS="${STEPS:-32}"
BIN="${PLOWRT_BIN:-$W/target/release/plowrt}"
LEASE="${PLOW_GPULEASE_BIN:-$W/perf-data/tools/gpulease}"
BASE_ASSETS="${L1_BASE_ASSETS:-/home/lava/plow/build-amd/l1-basepkt}"
D1_ASSETS="${L1_D1_ASSETS:-/home/lava/plow/build-amd/l1-d1}"
D8_ASSETS="${L1_D8_ASSETS:-/home/lava/plow/build-amd/l1-d8}"
BASE_HSACO="${L1_BASE_HSACO:-/home/lava/plow/build-amd/l1-base}"
HSACO="${L1_HSACO:-/home/lava/plow/build-amd/l1-gfx950}"
# A MEANINGFUL prompt, not random ids. A random-id prompt makes the model emit one
# constant token forever, and a constant stream would compare equal even if the merge
# were subtly wrong — the gate has to be able to fail.
PROMPT="${PROMPT:-$(python3 "$W/scripts/l1_prompt.py")}"

one() { # <label> <blob-dir> <hsaco-dir>
  local label=$1 assets=$2 hsaco=$3 log json ids
  log=$(mktemp); json=$(mktemp); ids="/tmp/l1-tok-$label.txt"
  "$LEASE" -n 1 "l1-tok-$1" \
      env PLOW_HSACO="$hsaco" PLOW_CHECKPOINT="$CK" \
      "$BIN" bench --assets "$assets" \
      --prompt-ids "$PROMPT" --concurrency 1 --requests 1 \
      --warmup-requests 0 --output-len "$STEPS" --token-audit \
      >"$json" 2>"$log" || { cat "$log"; rm -f "$log" "$json"; exit 1; }
  python3 - "$json" "$ids" "$label" "$assets" "$hsaco" "$CK" "$PROMPT" "$STEPS" <<'PY' || {
import json, os, sys

report_path, ids_path, label, assets, hsaco, checkpoint, prompt, steps = sys.argv[1:]
with open(report_path) as f:
    report = json.load(f)

def need(ok, message):
    if not ok:
        raise SystemExit(f"{label}: {message}")

expected_prompt = [int(token.strip()) for token in prompt.split(",")]
audit = report.get("token_audit") or {}
prompts = audit.get("prompt_token_ids")
outputs = audit.get("output_token_ids")
need(report.get("schema") == "plowrt.bench.v1", "unexpected bench schema")
need(report.get("vendor") == "Some(Amd)", "production bench did not use AMD")
need(report.get("requests") == 1 and report.get("completed") == 1 and report.get("failed") == 0,
     "request did not complete exactly once")
need(report.get("warmup_requests") == 0, "unexpected warmup request")
need(prompts == [expected_prompt], "token audit did not preserve the exact prompt")
need(isinstance(outputs, list) and len(outputs) == 1, "token audit has no unique output row")
need(len(outputs[0]) == int(steps), "token audit returned an incomplete stream")
need(any(outputs[0]), "all-zero output makes token identity vacuous")

artifacts = report.get("artifacts") or {}
packet = (artifacts.get("packet") or {}).get("path")
ckpt = (artifacts.get("checkpoint") or {}).get("path")
objects = artifacts.get("object_inventory") or []
need(os.path.realpath(report.get("asset_dir", "")) == os.path.realpath(assets),
     "bench loaded a different asset directory")
need(os.path.realpath(packet or "") == os.path.realpath(os.path.join(assets, "model.pkt")),
     "bench measured a different packet")
need(os.path.realpath(ckpt or "") == os.path.realpath(checkpoint),
     "bench bound a different checkpoint")
need(any(os.path.commonpath([os.path.realpath(obj.get("path", "")), os.path.realpath(hsaco)])
         == os.path.realpath(hsaco) for obj in objects),
     "bench did not inventory the selected code-object directory")

with open(ids_path, "w") as f:
    f.write(",".join(map(str, outputs[0])) + "\n")
PY
    cat "$log"
    rm -f "$log" "$json" "$ids"
    exit 1
  }
  echo "$label: $(head -c 120 "$ids")..."
  rm -f "$log" "$json"
}

one base "$BASE_ASSETS" "$BASE_HSACO"
one d1   "$D1_ASSETS"   "$HSACO"
one d8   "$D8_ASSETS"   "$HSACO"

fail=0
cmp -s /tmp/l1-tok-base.txt /tmp/l1-tok-d1.txt && echo "d1 == base  TOKEN-IDENTICAL" || { echo "d1 != base  MISMATCH"; fail=1; }
cmp -s /tmp/l1-tok-base.txt /tmp/l1-tok-d8.txt && echo "d8 == base  TOKEN-IDENTICAL" || { echo "d8 != base  MISMATCH"; fail=1; }
exit $fail
