#!/usr/bin/env bash
# k3_batch_gate.sh — the correctness gate batched K3 decode must pass BEFORE its refusals are
# lifted.                                                                    [K3-BATCH-GATE]
#
# WRITE THE GATE FIRST. `perf-data/archive/k3/k3-batched-decode-design.md` §5/§6.2 puts this ahead of the
# remaining wiring on purpose: the KDA recurrence is the model's core, a wrong one is FLUENT
# rather than broken, and the two refusals it would replace
# (`exec/amd.rs:3264`, `serve/engine.rs:187`) are today the only thing standing between a
# half-wired batch path and silent cross-sequence corruption. A gate that exists after the
# refusals are lifted is a gate that arrived too late.
#
# THE TWO CHECKS, and they catch different bugs:
#
#   A. IDENTICAL PROMPTS. B copies of one prompt must produce B IDENTICAL streams. Catches a
#      shared carried state directly: if slot 1's token threads into slot 0's KDA state the
#      streams diverge, and because every slot ran the same prompt any difference at all is a
#      bug rather than a legitimate difference.
#
#   B. DIFFERENT PROMPTS, RAGGED LENGTHS, COMPARED ACROSS TWO BATCH WIDTHS. The same B different
#      prompts are run at width B and at a SECOND width, and slot s must agree between them.
#      Catches per-slot position and kvlen handling, which check A cannot see because identical
#      prompts share their positions. Lengths are deliberately unequal —
#      `perf-data/batched-decode-amd-status.md:19-31` is the precedent, where exactly this shape
#      (prompts of length 3/5/7/4) caught ragged-position bugs on the dense path.
#
#      IT COMPARES TWO BATCHED RUNS, NOT A BATCHED RUN AGAINST A SOLO ONE, and that is a
#      correction rather than a convenience. A batched decode routes MoE through the GROUPED
#      expert kernel and a B=1 decode through the per-slot one; they accumulate in different
#      orders, and greedy decoding turns any tie-break into a different token a few steps later.
#      Measured: B=1 continues "The population is approximately 67 million people", B=4/8/16 all
#      continue "The capital of Germany is Berlin" — both fluent, both right, neither a defect.
#      Token-identity across those two paths is a criterion no correct implementation can meet,
#      so demanding it made the gate report FAIL on a working batch. Two batched widths DO share
#      a kernel, so between them token-identity is exactly the right bar — and it still tests
#      what check B is for, because the per-slot strides, positions and kvlens differ with width.
#
# Check B is the one that matters most and the one most likely to be skipped, because it needs
# two separately compiled production engines and therefore two full model loads rather than one.
#
#   ./scripts/k3_batch_gate.sh <assets-dir> <hsaco-dir> <checkpoint> [B] [alt-assets] [alt-hsaco]
#
# `alt-blob`/`alt-hsaco` are a build at a DIFFERENT batch width (both must move together — the
# hsaco carries PLOW_DECODE_BATCH too, since it sizes PLOW_GEMV_MM). Check B is skipped, loudly,
# without them.
#
# THE ALT BUILD MUST BE A DIFFERENT WIDTH, and reusing $BLOB for it compares the batch path
# against ITSELF at the same width — which passes whenever the batch path is self-consistently
# wrong.
#
#   STEPS  decode steps per arm (default 24)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOB="${1:?blob dir}"; HSACO="${2:?hsaco dir}"; CKPT="${3:?checkpoint}"; B="${4:-4}"
ALT_BLOB="${5:-}"; ALT_HSACO="${6:-}"
STEPS="${STEPS:-24}"
BIN="${PLOWRT_BIN:-$WT/target/release/plowrt}"
LEASE="${PLOW_GPULEASE_BIN:-$WT/perf-data/tools/gpulease}"
NIX="${PLOW_NIX_BIN:-nix}"

# Four prompts of DELIBERATELY UNEQUAL length. Token ids, not text: `--prompt` takes ids.
P1="1008,10484,318,15383,387"          # 5 — the README's known-good "capital of France"
P2="1008,10484,318"                    # 3
P3="1008,10484,318,15383,387,13,646"   # 7
P4="1008,10484,318,15383"              # 4
PROMPTS=("$P1" "$P2" "$P3" "$P4")

batch_capacity() { # <assets>
  python3 - "$1/build.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    value = json.load(source).get("shapes", {}).get("decode_batch")
if not isinstance(value, int) or value < 1:
    raise SystemExit(f"missing positive shapes.decode_batch in {sys.argv[1]}")
print(value)
PY
}

run() { # <label> <assets> <hsaco> <expected-capacity> <prompt>...
  local lbl="$1"; shift; local assets="$1"; shift; local hsaco="$1"; shift
  local capacity="$1"; shift; local log="/tmp/k3bg_$lbl.log" report rows ids cmd
  report=$(mktemp); rows=$(mktemp); ids="/tmp/k3bg_$lbl.ids"
  printf '%s\n' "$@" >"$rows"
  printf -v cmd \
    'exec %q --rt-checkpoint %q --rt-hsaco %q bench --assets %q --prompt-rows %q --concurrency %q --requests %q --warmup-requests 0 --output-len %q --token-audit --engine-diagnostics --max-hold-ms 8 --slo-ms 60000 >%q' \
    "$BIN" "$CKPT" "$hsaco" "$assets" "$rows" "$capacity" "$capacity" "$STEPS" "$report"
  GPU_LEASE_TIMEOUT=7200 "$LEASE" -n 8 "k3bg-$lbl" sg render -c \
    "PLOW_L2_PLACE_DISPATCH=1 $(printf %q "$NIX") develop '$WT' --command bash -c $(printf %q "$cmd")" \
    >"$log" 2>&1 || { cat "$log"; rm -f "$report" "$rows" "$ids"; return 1; }
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
need(report.get("vendor") == "Some(Amd)" and report.get("num_gpus") == 8,
     "gate did not run the AMD TP8 production engine")
need((report.get("requests"), report.get("completed"), report.get("failed")) == (capacity, capacity, 0),
     "incomplete production-mux requests")
need(report.get("warmup_requests") == 0, "unexpected warmup requests")
need(audit.get("prompt_token_ids") == prompts, "token audit reordered or changed prompt rows")
need(isinstance(outputs, list) and len(outputs) == capacity and
     all(isinstance(row, list) and len(row) == steps for row in outputs),
     "missing or incomplete token-audit output rows")
need(any(token != 0 for row in outputs for token in row), "all generated tokens are zero")
inp = report.get("input") or {}
lengths = list(map(len, prompts))
need(inp.get("mode") == "token_rows" and inp.get("row_count") == capacity and
     inp.get("min_tokens_per_request") == min(lengths) and
     inp.get("max_tokens_per_request") == max(lengths), "untruthful ragged input report")
engine = report.get("engine") or {}
need(engine.get("batch_capacity") == capacity, "loaded engine batch capacity differs from build")
diag = report.get("diagnostics") or {}
need(diag.get("supported") is True and diag.get("complete") is True and
     diag.get("overflowed") is False, "missing or partial engine diagnostics")
need(any(row.get("occupied_rows") == capacity and row.get("bucket", 0) >= capacity
         for row in diag.get("decode_selections", [])), "no full-width decode dispatch")
artifacts = report.get("artifacts") or {}
packet = (artifacts.get("packet") or {}).get("path")
packet_identity = (artifacts.get("packet") or {}).get("checksum")
checkpoint_info = artifacts.get("checkpoint") or {}
bound = checkpoint_info.get("path")
objects = artifacts.get("object_inventory") or []
real = os.path.realpath
need(real(report.get("asset_dir", "")) == real(assets), "bench loaded a different asset directory")
need(real(packet or "") == real(os.path.join(assets, "model.pkt")), "bench loaded a different packet")
need(isinstance(packet_identity, str) and packet_identity, "packet identity is missing")
need(real(bound or "") == real(checkpoint), "bench bound a different checkpoint")
need(isinstance(checkpoint_info.get("layout_checksum"), str) and checkpoint_info["layout_checksum"],
     "checkpoint identity is missing")
need(any(os.path.commonpath([real(obj.get("path", "")), real(hsaco)]) == real(hsaco) and
         isinstance(obj.get("checksum"), str) and obj["checksum"]
         for obj in objects), "bench did not identify the selected object directory")
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

[[ "$B" =~ ^[0-9]+$ ]] && [ "$B" -ge 2 ] || { echo "B must be an integer >= 2" >&2; exit 2; }
PRIMARY_CAP=$(batch_capacity "$BLOB") || exit 1
[ "$PRIMARY_CAP" -eq "$B" ] || { echo "primary build capacity $PRIMARY_CAP != requested B=$B" >&2; exit 1; }

echo "=== CHECK A: $B copies of one prompt must give $B identical streams ==="
IDENTICAL=(); for _ in $(seq 1 "$B"); do IDENTICAL+=("$P1"); done
if ! run "identical" "$BLOB" "$HSACO" "$B" "${IDENTICAL[@]}"; then
  echo "  decode command failed — see /tmp/k3bg_identical.log"
  echo ">>> CHECK A: FAIL"; A_OK=1
else
  mapfile -t A </tmp/k3bg_identical.ids
  uniq_n=$(printf '%s\n' "${A[@]}" | sort -u | wc -l)
  printf '  %s\n' "${A[@]}"
  if ! printf '%s\n' "${A[@]}" | grep -Eq '[1-9]'; then
    echo ">>> CHECK A: FAIL — every generated token is zero; identical dead streams are not correctness"
    A_OK=1
  elif [ "$uniq_n" -eq 1 ]; then
    echo ">>> CHECK A: PASS ($B identical)"; A_OK=0
  else
    echo ">>> CHECK A: FAIL — $uniq_n distinct streams; a slot is reading another's state"; A_OK=1
  fi
fi

echo
echo "=== CHECK B: $B ragged prompts must give the same streams at a second width ==="
if [ -z "$ALT_BLOB" ] || [ -z "$ALT_HSACO" ]; then
  # Silently reusing $BLOB here would compare the batch path against itself and print PASS.
  echo "  NO ALT BUILD GIVEN (arguments 5 and 6). Check B needs a build at a DIFFERENT batch"
  echo "  width; without one it would compare the batched blob to itself and pass vacuously."
  echo ">>> CHECK B: NOT RUN"
  B_OK=1
else
B_OK=0
ALT_CAP=$(batch_capacity "$ALT_BLOB") || B_OK=1
if [ "$B_OK" -eq 0 ] && [ "$ALT_CAP" -eq "$B" ]; then
  echo "  alt build has the same capacity $B; comparison would be vacuous"; B_OK=1
fi
if [ "$B_OK" -eq 0 ] && [ "$ALT_CAP" -lt 2 ]; then
  echo "  alt build capacity $ALT_CAP leaves fewer than two comparable slots"; B_OK=1
fi
cycle_prompts "$B"; PRIMARY_PROMPTS=("${CYCLED[@]}")
cycle_prompts "${ALT_CAP:-0}"; ALT_PROMPTS=("${CYCLED[@]}")
if [ "$B_OK" -eq 0 ] && ! run "ragged" "$BLOB" "$HSACO" "$B" "${PRIMARY_PROMPTS[@]}"; then B_OK=1; fi
if [ "$B_OK" -eq 0 ] && ! run "ragged_alt" "$ALT_BLOB" "$ALT_HSACO" "$ALT_CAP" "${ALT_PROMPTS[@]}"; then B_OK=1; fi
if [ "$B_OK" -ne 0 ]; then
  echo "  a decode command failed — see /tmp/k3bg_ragged{,_alt}.log"; B_OK=1
else
  mapfile -t BATCHED </tmp/k3bg_ragged.ids
  mapfile -t ALT </tmp/k3bg_ragged_alt.ids
  n=$(( B < ALT_CAP ? B : ALT_CAP ))
  echo "  comparing $n slots ($B-wide vs ${#ALT[@]}-wide)"
  for i in $(seq 0 $((n-1))); do
    if [ "${BATCHED[$i]}" = "${ALT[$i]}" ]; then echo "  slot $i: MATCHES the other width"
    else echo "  slot $i: DIFFERS across widths"; echo "     w=$B   ${BATCHED[$i]:0:70}"; echo "     w=${#ALT[@]} ${ALT[$i]:0:70}"; B_OK=1; fi
  done
fi
fi
[ "$B_OK" -eq 0 ] && echo ">>> CHECK B: PASS" || echo ">>> CHECK B: FAIL — per-slot position/kvlen handling differs with batch width"

echo
if [ "${A_OK:-1}" -eq 0 ] && [ "$B_OK" -eq 0 ]; then
  echo "BATCH GATE: PASS at B=$B — the refusals may be lifted"
  exit 0
fi
echo "BATCH GATE: NOT PASSED — leave exec/amd.rs:3264 and serve/engine.rs:187 in place"
exit 1
