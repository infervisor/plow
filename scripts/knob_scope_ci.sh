#!/usr/bin/env bash
# Checkpoint S for every knob a change touches (plans/lean-knob-verification.md §4.5).
#
#   scripts/knob_scope_ci.sh <base-ref> [plowc] [plowrt]
#
# A knob is touched when a changed line (against <base-ref>) is its registry line or names its
# env var. Each touched knob runs `plowc knob-scope` on the production recipe and on its one-layer
# cut (PLOW_LAYERS=1), and a rejection fails the run. Values: a bool knob flips its default; any
# other knob takes KNOB_SCOPE_VALUES="emit.id=OFF,ON;rt.id=OFF,ON" and fails without one. Runtime
# knobs replay KNOB_SCOPE_WORKLOAD through the route trace.
#
# Environment:
#   KNOB_SCOPE_HF_DIR   the production checkpoint (e.g. /workspace/models/GLM-5.3-plow-lite)
#   KNOB_SCOPE_RECIPE   a build.json whose emit_config.replay is the production recipe
#   PLOW_VERIFY_BIN     a plow_verify that implements checkpoints K and S
#   KNOB_SCOPE_VALUES   test values for non-bool knobs: "emit.id=OFF,ON;rt.id=OFF,ON"
#   KNOB_SCOPE_WORKLOAD route workload JSON, for runtime knobs
#   KNOB_SCOPE_OUT      output directory (default knob-scope-ci)
# The first three must be set for any knob to be checked. Unset on a change that touches knobs,
# the run emits a GitHub warning naming the unchecked knobs and the missing variables and exits 0;
# set, every rejection is fatal.
set -euo pipefail
base=${1:?usage: knob_scope_ci.sh <base-ref> [plowc] [plowrt]}
plowc=${2:-target/release/plowc}
plowrt=${3:-target/release/plowrt}
out=${KNOB_SCOPE_OUT:-knob-scope-ci}

touched=$(git diff -U0 "$base" -- crates | python3 -c '
import re, sys
changed = [l[1:] for l in sys.stdin if l[:1] in "+-" and not l.startswith(("+++", "---"))]
spec = re.compile(r"KnobSpec::new\(\"([a-z]+\.[A-Za-z0-9_]+)\", (?:Some\(\"([A-Z0-9_]+)\"\)|None), Layer::(\w+), ([^,]+), ([^,]+),")
table = {}
for path in ("crates/devgen/src/knob_spec.rs", "crates/plowrt/src/knob_spec.rs"):
    for line in open(path):
        m = spec.search(line)
        if m and m.group(3) in ("Emit", "Runtime"):
            table[m.group(1)] = (m.group(2), m.group(4).strip(), m.group(5).strip())
hit = set()
for line in changed:
    for id_, (env, _, _) in table.items():
        if f"\"{id_}\"" in line or (env and re.search(rf"\b{env}\b", line)):
            hit.add(id_)
for id_ in sorted(hit):
    env, domain, default = table[id_]
    print(id_, domain, default)
')

if [ -z "$touched" ]; then
    echo "knob-scope: no knob touched since $base"
    exit 0
fi
missing=()
for var in KNOB_SCOPE_HF_DIR KNOB_SCOPE_RECIPE PLOW_VERIFY_BIN; do
    [ -n "${!var:-}" ] || missing+=("$var")
done
if [ ${#missing[@]} -gt 0 ]; then
    knobs=$(cut -d' ' -f1 <<< "$touched" | paste -sd, -)
    echo "::warning title=knob scope not checked::checkpoint S did not run for ${knobs}: ${missing[*]} not set on this runner"
    exit 0
fi
emit_args=(--hf-dir "$KNOB_SCOPE_HF_DIR" --gpu MI300X --num-gpus 8 --batch 1,4,8,16,20
    --emit-decode-batch-ladder 1,2,4,8,16,20 --seq 512,2048,8192 --max-ctx 81920 --arch gfx942
    --emit devblob --replay-knobs "$KNOB_SCOPE_RECIPE")

values_for() {
    local id=$1 domain=$2 default=$3 pair
    for pair in ${KNOB_SCOPE_VALUES//;/ }; do
        [ "${pair%%=*}" = "$id" ] && { echo "${pair#*=}"; return; }
    done
    case "$domain:$default" in
        Domain::Bool:ON) echo "1,0" ;;
        Domain::Bool:*) echo ",1" ;;
        *) return 1 ;;
    esac
}

fail=0
while read -r id domain default; do
    if ! values=$(values_for "$id" "$domain" "$default"); then
        echo "knob-scope: $id is not a bool; give its test values in KNOB_SCOPE_VALUES" >&2
        fail=1
        continue
    fi
    extra=()
    case "$id" in
        rt.*) extra=(--workload "${KNOB_SCOPE_WORKLOAD:?$id is a runtime knob: set KNOB_SCOPE_WORKLOAD}") ;;
    esac
    for layers in all 1; do
        dir="$out/$id-$layers"
        echo "== knob-scope $id values=$values layers=$layers"
        if ! env ${layers:+PLOW_LAYERS=$layers} "$plowc" "${emit_args[@]}" knob-scope \
            --knob "$id" --values "$values" --plowrt "$plowrt" --dir "$dir" "${extra[@]}"; then
            fail=1
        fi
    done
done <<< "$touched"
exit $fail
