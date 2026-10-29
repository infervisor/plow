#!/usr/bin/env bash
# plowbench-doctor — validate the environment and the artifacts BEFORE leasing a GPU.
#
#   nix develop --command scripts/bench/plowbench-doctor.sh [assets-dir] [object-dir] [plowrt] [arch] [serve|block]
#   (block skips the vLLM client check)
#
# CPU only. Leases nothing, starts no server, touches no GPU. Run it first; every check here
# corresponds to a failure that has already cost at least one leased run in this campaign.
# [8] is the capability inventory: a packet route that is off and not waived in the recipe's
# [capabilities] table (or PB_CAPABILITY_WAIVERS) fails, so a campaign starts from the full set.
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
# plowrt binds the first loadable cuBLASLt: beside the binary (<exe>/../lib, <exe>), then the
# loader's libcublasLt.so.13/.12/.so (crates/plowrt/src/device/cuda/lt.rs). A packet whose
# build.json declares runtime_requires.cublaslt refuses an older one at load, inside the lease.
# Run the doctor with the serve environment: LD_LIBRARY_PATH decides the loader's pick.
if pb_is_nvidia "$ARCH" && [ -x "$PLOWRT" ]; then
    lt=$(python3 -I - "$PLOWRT" "${ASSETS:-}" <<'PYEOF'
import ctypes, json, os, sys
d = os.path.dirname(os.path.realpath(sys.argv[1]))
names = ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]
cands = [p for x in (os.path.join(d, "..", "lib"), d) for p in (os.path.join(x, n) for n in names)
         if os.path.isfile(p)] + names
got = None
for c in cands:
    try:
        got = (c, ctypes.CDLL(c).cublasLtGetVersion())
        break
    except (OSError, AttributeError):
        continue
bj = os.path.join(sys.argv[2], "build.json") if sys.argv[2] else ""
need = (json.load(open(bj)).get("runtime_requires") or {}).get("cublaslt") if os.path.isfile(bj) else None
if got is None:
    print("FAIL\tno loadable libcublasLt beside plowrt or on the loader path")
    sys.exit()
v = got[1]
have = f"{v // 10000}.{v // 100 % 100}.{v % 100}"
if need:
    maj, mnr = (int(x) for x in str(need).split(".")[:2])
    if v < maj * 10000 + mnr * 100:
        print(f"FAIL\tplowrt binds cuBLASLt {have} ({got[0]}); the packet needs >= {need}: "
              f"put that libcublasLt.so.13 in {os.path.normpath(os.path.join(d, '..', 'lib'))} "
              "or first on LD_LIBRARY_PATH")
        sys.exit()
print(f"ok\tcuBLASLt {have} ({got[0]})" + (f", packet needs >= {need}" if need else ""))
PYEOF
)
    case "$lt" in
        ok*) pb_ok "${lt#ok	}" ;;
        FAIL*) pb_bad "${lt#FAIL	}" ;;
        *) pb_warn "cuBLASLt check: $lt" ;;
    esac
fi
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
    # Lean qualification (`campaign.py build` writes it; `campaign.py qualify --assets` refreshes).
    LQ="$ASSETS/../lean-qualification.json"
    if [ -f "$LQ" ]; then
        lq=$(python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
if not d.get("checked"): print("unchecked", d.get("reason","")[:120]); sys.exit()
p=d["packets"]; bad=[e for e in p if not e["qualified"]]
print("ok" if not bad else "gaps", f"{len(p)-len(bad)}/{len(p)} packets qualify;", "; ".join(e["packet"].rsplit("/",1)[-1]+": "+e["qualification"]["gaps"][0][:100] for e in bad[:3]))' "$LQ")
        case "$lq" in
            ok*) pb_ok "lean qualification: ${lq#ok }" ;;
            *) pb_warn "lean qualification: $lq (structural evidence only; PLOW_LEAN_QUALIFY=strict would refuse)" ;;
        esac
    else
        pb_warn "no lean-qualification.json beside $ASSETS — run campaign.py qualify --assets $ASSETS"
    fi
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
    pb_warn "no gpulease at $WT/perf-data/tools/gpulease, /app/plow/perf-data/tools/gpulease or on PATH"
fi
QROOT="${PB_GPUQ:-}"
if [ -n "$QROOT" ] && [ -e "$QROOT/runner.log" ]; then
    # The lab submit.sh queue runs runner.py; scripts/bench/gpuq.py runs `gpuq.py --root <root> run`.
    if pgrep -f "$QROOT/runner.py" > /dev/null 2>&1 \
        || pgrep -f "gpuq.py --root $(realpath -m "$QROOT") run" > /dev/null 2>&1; then
        pb_ok "queue runner alive ($(tail -1 "$QROOT/runner.log" | cut -c1-80))"
    else
        pb_warn "queue runner NOT running — it idle-exits after 1800 s on an empty spool."
        pb_info "lab submit.sh: a job submitted after it exits sits in the spool forever; restart it first."
        pb_info "gpuq.py submit restarts its runner itself."
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
echo "[8] capability inventory (routes the packet declares; an OFF needs a recipe waiver)"
# plowrt inventory is static (CPU): buckets, ladder, live KV, packed prefill + compact terminal,
# segment roles, token batch, generated kernels, sampler, Lt prefill. A campaign starts from the
# full set and measures removing routes; an unwaived OFF is a route it would rediscover later.
if [ -n "$ASSETS" ] && [ -x "$PLOWRT" ]; then
    while IFS=$'\t' read -r lvl msg; do
        case "$lvl" in
            ok) pb_ok "$msg" ;;
            WARN) pb_warn "$msg" ;;
            FAIL) pb_bad "$msg" ;;
            *) pb_info "$lvl $msg" ;;
        esac
    done < <(python3 "$HERE/capability_check.py" "$PLOWRT" "$ASSETS" 2>&1)
else
    pb_warn "capability inventory skipped (needs the assets dir and an executable plowrt)"
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
