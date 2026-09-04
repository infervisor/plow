#!/usr/bin/env bash
# Stack-2 bundles from the promoted-default source: re-key the grouped-MoE route records to the
# new emitter label, then build the served bundle (/tmp/k3-stack2) and the exact-arm bundle
# (/tmp/k3-stack2-exact, PLOW_ATTNRES_F32MIX=0) for the checksum gate.
set -uo pipefail
wt=/home/lava/plow/.claude/worktrees/d1-moe-decode-rule; out=/tmp/k3-xr-phase-gate
S=/home/lava/.claude/jobs/3cf96348/tmp
cd "$wt"
rm -rf /tmp/k3-stack2 /tmp/k3-stack2-exact
"$S/showdown_bundle.sh" /tmp/k3-stack2 > "$out/stack2-bundle1.log" 2>&1
label=$(grep -o -E "want gfx950-[0-9a-f]{16}" /tmp/k3-stack2/emit.log | head -1 | cut -d' ' -f2)
if [ -n "$label" ]; then
  echo "[$(date +%T)] rekey to $label"
  rm -f tuning/amd/gfx950/mi350x/moe_decode_measurement.jsonl
  python3 scripts/tune_moe_decode_publish.py --root tuning --hardware amd/gfx950/mi350x --n-cu 256 --rung 1 \
    --topk 16 --hidden 3584 --inter-local 384 --experts 896 --enc mxfp4 --digest "$label" \
    --toolchain rocm-7.14.0-nix --campaign k3-moe-decode-network-derived-20260904 \
    --interpreter-us $(cat "$out/d1-interp-samples.txt") --standalone-us $(cat "$out/d1-standalone-samples.txt")
  git add tuning/amd/gfx950/mi350x/moe_decode_measurement.jsonl
  git commit -q -m "tunedb: re-key K3 grouped-MoE decode route records to the stack-2 source label

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FEN7AT33rmNdwAoePSheAX" || true
  rm -rf /tmp/k3-stack2
  "$S/showdown_bundle.sh" /tmp/k3-stack2 > "$out/stack2-bundle2.log" 2>&1
else
  echo "[$(date +%T)] records already keyed to the current label"
fi
grep -c "want gfx950" /tmp/k3-stack2/emit.log
grep -E "^lean:|^tiles:|^objects:" "$out/stack2-bundle2.log" "$out/stack2-bundle1.log" 2>/dev/null | tail -3
grep -oE "route=[A-Za-z]+|MoeDecodeRoute::[A-Za-z]+|selected (standalone|interpreter)[^\n]{0,60}" /tmp/k3-stack2/emit.log | sort | uniq -c | head -3
"$S/showdown_bundle.sh" /tmp/k3-stack2-exact PLOW_ATTNRES_F32MIX=0 > "$out/stack2-exact-bundle.log" 2>&1
grep -E "^lean:|^tiles:|^objects:" "$out/stack2-exact-bundle.log"
echo "f32mix objects: served=$(ls /tmp/k3-stack2/hsaco | grep -c f32mix) exact=$(ls /tmp/k3-stack2-exact/hsaco | grep -c f32mix); regstate: served=$(ls /tmp/k3-stack2/hsaco | grep -c regstate) exact=$(ls /tmp/k3-stack2-exact/hsaco | grep -c regstate)"
echo "[$(date +%T)] STACK2_PREP_DONE"
