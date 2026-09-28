#!/usr/bin/env bash
# plowbench-doctor — validate the environment and the artifacts BEFORE leasing a GPU.
#
#   nix develop --command scripts/bench/plowbench-doctor.sh [assets-dir] [object-dir] [plowrt]
#
# CPU only. Leases nothing, starts no server, touches no GPU. Run it first; every check here
# corresponds to a failure that has already cost at least one leased run in this campaign.
#
# Exit 0 = safe to lease. Exit 1 = something will fail after the weights load. Exit 2 = warnings
# only (the run will work but the number may not mean what you think).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/plowbench.sh"

ASSETS="${1:-${PB_ASSETS:-}}"
OBJDIR="${2:-${PLOW_HSACO:-}}"
PLOWRT="${3:-${PLOWRT_BIN:-$WT/target/release/plowrt}}"
ARCH=$(pb_detect_arch "${4:-${PB_ARCH:-${TARGET_ARCH:-}}}" "$ASSETS" "$OBJDIR")

echo "plowbench-doctor  $(date -u +%FT%TZ)"
echo "  worktree:    $WT"
echo "  target arch: $ARCH"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[ \t]*//')
    [ -n "$gpu_name" ] && echo "  device:      $gpu_name (NVIDIA)"
elif command -v rocm-smi >/dev/null 2>&1; then
    echo "  device:      ROCm device"
fi
echo

echo "[1] environment"
pb_require_nix
if command -v python3 > /dev/null 2>&1; then pb_ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)"
else pb_bad "no python3"; fi
if command -v curl > /dev/null 2>&1; then pb_ok "curl"; else pb_bad "no curl (readiness poll needs it)"; fi

echo
echo "[2] measurement hazards in the current environment"
pb_hazard_env
[ "$PB_WARN" = 0 ] && pb_ok "no hazardous PLOW_* overrides set"

echo
echo "[3] binaries"
pb_check_plowrt "$PLOWRT" "$ARCH"
if [ "${5:-serve}" != block ]; then
    pb_check_vllm "$ARCH"
fi
# plowc is needed only for emit, and lives in a per-campaign target dir more often than not.
for c in "$WT/target/release/plowc" "$WT/target-glm53/release/plowc"; do
    [ -x "$c" ] && { pb_ok "plowc $c"; break; }
done

echo
echo "[4] artifacts"
if [ -n "$ASSETS" ]; then
    pb_check_assets "$ASSETS" "${PB_PACKET_SHA16:-}"
else
    pb_warn "no assets dir given — pass one, or set PB_ASSETS, to check the packet"
fi
if [ -n "$OBJDIR" ]; then
    pb_check_objects "$OBJDIR" "$ARCH" "$ASSETS"
else
    pb_warn "no object dir given — pass one, or set PLOW_HSACO, to check the object set"
fi

echo
echo "[5] GPUs and the queue"
LEASE=""
for l in "$WT/perf-data/tools/gpulease" /app/plow/perf-data/tools/gpulease; do
    if [ -x "$l" ]; then LEASE="$l"; break; fi
done
if [ -z "$LEASE" ] && command -v gpulease >/dev/null 2>&1; then
    LEASE="$(command -v gpulease)"
fi
if [ -x "$LEASE" ]; then
    pb_ok "gpulease at $LEASE"
    # Print its own words; do not invent a card count from them.
    "$LEASE" --status 2>/dev/null | head -4 | sed 's/^/        /'
else
    pb_warn "gpulease not at $LEASE — it is NOT on PATH; every GPU process must go through it"
fi
QROOT="${PB_GPUQ:-}"
if [ -n "$QROOT" ] && [ -e "$QROOT/runner.log" ]; then
    if pgrep -f "$QROOT/runner.py" > /dev/null 2>&1; then
        pb_ok "queue runner alive ($(tail -1 "$QROOT/runner.log" | cut -c1-80))"
    else
        pb_warn "queue runner NOT running — it idle-exits after 1800 s on an empty spool."
        pb_info "A job submitted after it exits sits in the spool forever. Restart it, then submit."
    fi
fi

echo
echo "[6] scratch space"
for d in /workspace /; do
    use=$(df -P "$d" 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%')
    [ -n "$use" ] || continue
    if [ "$use" -ge 95 ]; then pb_bad "$d is ${use}% full"
    elif [ "$use" -ge 85 ]; then pb_warn "$d is ${use}% full — put build output and exports on /workspace"
    else pb_ok "$d ${use}% used"; fi
done

echo
echo "[7] worktree vs HEAD (harness files, recipe)"
# A shared worktree whose HEAD moved (commits written from another index) keeps stale copies of the
# scripts and recipes; a run from it measures something other than HEAD says.
stale=$(git -C "$WT" status --porcelain -- recipes scripts/bench scripts/campaign scripts/voice scripts/llm scripts/tts 2>/dev/null)
if [ -n "$stale" ]; then
    pb_warn "$(echo "$stale" | wc -l) harness/recipe file(s) differ from HEAD $(git -C "$WT" rev-parse --short HEAD) in $WT:"
    echo "$stale" | head -6 | sed 's/^/          /'
    pb_info "stale copy or uncommitted edits: run from a detached worktree (git worktree add --detach <dir> HEAD)"
else
    pb_ok "harness scripts and recipes match HEAD $(git -C "$WT" rev-parse --short HEAD 2>/dev/null)"
fi
REC="${ASSETS:+$ASSETS/../build-record.json}"
if [ -n "$REC" ] && [ -f "$REC" ]; then
    python3 - "$REC" "$WT" <<'PYEOF'
import hashlib, json, os, subprocess, sys
r = json.load(open(sys.argv[1])); wt = sys.argv[2]
path = r.get("recipe", "")
rel = path.split("/recipes/", 1)[-1] if "/recipes/" in path else None
head = subprocess.run(["git", "-C", wt, "show", f"HEAD:recipes/{rel}"], capture_output=True).stdout if rel else b""
if not head:
    print(f"  WARN  build-record recipe {path} is not in HEAD's recipes/")
elif hashlib.sha256(head).hexdigest() != r.get("recipe_sha256"):
    print(f"  WARN  assets were built from a recipes/{rel} that differs from HEAD's (rebuild, or say why)")
else:
    print(f"  ok    assets built from HEAD's recipes/{rel}")
PYEOF
fi

echo
if [ "$PB_FAIL" -gt 0 ]; then
    echo "RESULT: $PB_FAIL failure(s), $PB_WARN warning(s) — DO NOT lease a GPU yet."
    exit 1
fi
if [ "$PB_WARN" -gt 0 ]; then
    echo "RESULT: clean, with $PB_WARN warning(s) — safe to lease, but read them first."
    exit 2
fi
echo "RESULT: all checks passed — safe to lease."
exit 0
