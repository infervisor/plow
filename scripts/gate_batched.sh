#!/usr/bin/env bash
# Production-mux correctness gate for batched Gemma decode on gfx950.
#
# (a) B copies of one prompt must produce B identical nonzero streams.
# (b) Ragged prompts at B=4/B=8 must reproduce their exact batch-1 streams.
set -euo pipefail

WT="${PLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CKPT="${CKPT:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
NIX="${PLOW_NIX_BIN:-nix}"
ASSETS_PREFIX="${G31B_ASSETS_PREFIX:-$WT/build-amd/g31b-db}"
HSACO_PREFIX="${G31B_HSACO_PREFIX:-$WT/build-amd/hsaco-b}"
STEPS="${STEPS:-4}"

P1="2,106,1645"
P2="2,106,1645,236764,3689"
P3="2,3689,506,7534,529,6427,236761"
P4="2,106,1645,236764"
PROMPTS=("$P1" "$P2" "$P3" "$P4")

run() { # <label> <capacity> <prompt>...
  local label=$1; shift; local capacity=$1; shift
  local assets="${ASSETS_PREFIX}${capacity}" hsaco="${HSACO_PREFIX}${capacity}"
  local report rows log ids cmd
  report=$(mktemp); rows=$(mktemp); log="/tmp/gate-batched-$label.log"
  ids="/tmp/gate-batched-$label.ids"
  printf '%s\n' "$@" >"$rows"
  printf -v cmd \
    'exec %q --rt-checkpoint %q --rt-hsaco %q bench --assets %q --prompt-rows %q --concurrency %q --requests %q --warmup-requests 0 --output-len %q --token-audit --engine-diagnostics --max-hold-ms 8 --slo-ms 60000 >%q' \
    "$BIN" "$CKPT" "$hsaco" "$assets" "$rows" "$capacity" "$capacity" "$STEPS" "$report"
  "$NIX" develop "$WT" --command bash -c "$cmd" >"$log" 2>&1 || {
    cat "$log"; rm -f "$report" "$rows" "$ids"; return 1
  }
  python3 - "$report" "$ids" "$rows" "$assets" "$hsaco" "$CKPT" "$capacity" "$STEPS" <<'PY' || {
import json, os, sys
report_path, ids_path, rows_path, assets, hsaco, checkpoint, capacity, steps = sys.argv[1:]
capacity, steps = int(capacity), int(steps)
with open(report_path) as source: report = json.load(source)
with open(rows_path) as source:
    prompts = [[int(token.strip()) for token in line.split(",")] for line in source.read().splitlines()]
def need(ok, message):
    if not ok: raise SystemExit(message)
audit = report.get("token_audit") or {}
outputs = audit.get("output_token_ids")
need(report.get("schema") == "plowrt.bench.v1", "unsupported bench report")
need(report.get("vendor") == "Some(Amd)", "gate did not use the AMD production engine")
need((report.get("requests"), report.get("completed"), report.get("failed")) == (capacity, capacity, 0),
     "incomplete production-mux requests")
need(report.get("warmup_requests") == 0, "unexpected warmup requests")
need(audit.get("prompt_token_ids") == prompts, "token audit reordered or changed prompt rows")
need(isinstance(outputs, list) and len(outputs) == capacity and
     all(isinstance(row, list) and len(row) == steps for row in outputs),
     "missing or incomplete token-audit output rows")
need(any(token != 0 for row in outputs for token in row), "all generated tokens are zero")
lengths = list(map(len, prompts)); inp = report.get("input") or {}
need(inp.get("mode") == "token_rows" and inp.get("row_count") == capacity and
     inp.get("min_tokens_per_request") == min(lengths) and
     inp.get("max_tokens_per_request") == max(lengths), "untruthful ragged input report")
need((report.get("engine") or {}).get("batch_capacity") == capacity,
     "loaded engine batch capacity differs from requested width")
diag = report.get("diagnostics") or {}
need(diag.get("supported") is True and diag.get("complete") is True and
     diag.get("overflowed") is False, "missing or partial engine diagnostics")
need(any(row.get("occupied_rows") == capacity and row.get("bucket", 0) >= capacity
         for row in diag.get("decode_selections", [])), "no full-width decode dispatch")
artifacts = report.get("artifacts") or {}; real = os.path.realpath
packet_info = artifacts.get("packet") or {}; checkpoint_info = artifacts.get("checkpoint") or {}
need(real(report.get("asset_dir", "")) == real(assets), "bench loaded a different asset directory")
need(real(packet_info.get("path", "")) == real(os.path.join(assets, "model.pkt")),
     "bench loaded a different packet")
need(isinstance(packet_info.get("checksum"), str) and packet_info["checksum"],
     "packet identity is missing")
need(real(checkpoint_info.get("path", "")) == real(checkpoint),
     "bench bound a different checkpoint")
need(isinstance(checkpoint_info.get("layout_checksum"), str) and checkpoint_info["layout_checksum"],
     "checkpoint identity is missing")
need(any(os.path.commonpath([real(obj.get("path", "")), real(hsaco)]) == real(hsaco) and
         isinstance(obj.get("checksum"), str) and obj["checksum"]
         for obj in artifacts.get("object_inventory") or []),
     "bench did not identify the selected object directory")
with open(ids_path, "w") as sink:
    for row in outputs: sink.write(json.dumps(row, separators=(",", ":")) + "\n")
PY
    cat "$log"; rm -f "$report" "$rows" "$ids"; return 1
  }
  rm -f "$report" "$rows"
}

cycle_prompts() { # <count>
  local count=$1 i
  CYCLED=()
  for ((i=0; i<count; i++)); do CYCLED+=("${PROMPTS[$((i % ${#PROMPTS[@]}))]}"); done
}

echo "#### reference: each prompt alone on the batch-1 production engine"
for i in "${!PROMPTS[@]}"; do run "solo-$i" 1 "${PROMPTS[$i]}"; done

echo "#### B=4, four copies of one prompt"
run identical 4 "$P1" "$P1" "$P1" "$P1"
mapfile -t identical </tmp/gate-batched-identical.ids
[ "$(printf '%s\n' "${identical[@]}" | sort -u | wc -l)" -eq 1 ] || {
  echo "FAIL: identical prompts produced distinct streams" >&2; exit 1
}

echo "#### ragged batches reproduce batch-1 streams"
for width in 4 8; do
  cycle_prompts "$width"
  run "ragged-$width" "$width" "${CYCLED[@]}"
  mapfile -t batched <"/tmp/gate-batched-ragged-$width.ids"
  for ((i=0; i<width; i++)); do
    reference=$(sed -n '1p' "/tmp/gate-batched-solo-$((i % ${#PROMPTS[@]})).ids")
    [ "${batched[$i]}" = "$reference" ] || {
      echo "FAIL: width $width row $i differs from its batch-1 stream" >&2; exit 1
    }
  done
done

echo "PASS: identical and ragged B=4/B=8 production-mux streams are non-vacuous and exact"
