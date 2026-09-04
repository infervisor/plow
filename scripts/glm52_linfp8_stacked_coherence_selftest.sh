#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/plow-glm52-coherence-selftest.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/objects" "$TMP/checkpoint"
touch "$TMP"/{stk_base,stk_lfp8}.pkt "$TMP/objects/interp_decode.elf"
printf '{"arm":"stk_base"}\n' >"$TMP/stk_base.build.json"
printf '{"arm":"stk_lfp8"}\n' >"$TMP/stk_lfp8.build.json"
printf '2,3,5\n' >"$TMP/prompt_ids.txt"

cat >"$TMP/gpulease" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'acquired\n' >>"${GLM_COH_LEASE_COUNT:?}"
shift 3
exec env ROCR_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3 "$@"
SH
cat >"$TMP/plowrt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "${ROCR_VISIBLE_DEVICES:-}" == 0,1,2,3 && "${HIP_VISIBLE_DEVICES:-}" == 0,1,2,3 ]]
checkpoint= objects= assets= prompt= steps=
while (($#)); do
  case "$1" in
    --rt-checkpoint) checkpoint=$2; shift 2 ;;
    --rt-hsaco) objects=$2; shift 2 ;;
    --assets) assets=$2; shift 2 ;;
    --prompt-ids) prompt=$2; shift 2 ;;
    --output-len) steps=$2; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$assets" "$objects" "$checkpoint" "$prompt" "$steps" <<'PY'
import json, os, sys
assets, objects, checkpoint, prompt, steps = sys.argv[1:]
prompt = [int(token) for token in prompt.split(",")]
steps = int(steps)
agreement = None if os.environ.get("GLM_COH_SELFTEST_NO_AGREEMENT") else {
    "ranks": 4, "sampled_token_every": 1,
    "counter_audit_every_dispatch": True, "prefill_completion_all_ranks": True,
}
print(json.dumps({
    "schema": "plowrt.bench.v1", "vendor": "Some(Amd)", "num_gpus": 4, "parallel": "tp",
    "asset_dir": assets, "requests": 1, "completed": 1, "failed": 0,
    "warmup_requests": 0, "prompt_tokens": len(prompt), "output_tokens": steps,
    "scheduler": {"rejected": 0, "admit_shed": 0},
    "artifacts": {
        "packet": {"path": os.path.join(assets, "model.pkt")},
        "build_manifest": {
            "path": (os.path.join(assets, "wrong-build.json")
                     if os.environ.get("GLM_COH_SELFTEST_WRONG_BUILD_PATH")
                     else os.path.join(assets, "build.json")),
            "checksum": ("" if os.environ.get("GLM_COH_SELFTEST_EMPTY_BUILD_CHECKSUM")
                         else "fnv1a64:0123456789abcdef"),
        },
        "checkpoint": {"path": checkpoint},
        "object_inventory": [{"path": os.path.join(objects, "interp_decode.elf")}],
    },
    "diagnostics": {"supported": True, "complete": True, "overflowed": False,
                    "rank_agreement": agreement},
    "token_audit": {"prompt_token_ids": [prompt],
                    "output_token_ids": [list(range(7, 7 + steps))]},
}))
PY
SH
chmod +x "$TMP/gpulease" "$TMP/plowrt"

LEASE_COUNT="$TMP/lease-count"
: >"$LEASE_COUNT"
PLOW_AB_DIR="$TMP" PLOW_CKPT_Q="$TMP/checkpoint" PLOW_HSACO="$TMP/objects" \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  GLM_COH_LEASE_COUNT="$LEASE_COUNT" STEPS=4 TP=4 \
  bash "$ROOT/scripts/glm52_linfp8_stacked_coherence.sh" >"$TMP/pass.log"
grep -q 'TP4 EVERY-TOKEN AGREEMENT' "$TMP/pass.log"
[[ "$(wc -l <"$LEASE_COUNT")" -eq 1 ]]

: >"$LEASE_COUNT"
if GLM_COH_SELFTEST_NO_AGREEMENT=1 PLOW_AB_DIR="$TMP" \
  PLOW_CKPT_Q="$TMP/checkpoint" PLOW_HSACO="$TMP/objects" \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  GLM_COH_LEASE_COUNT="$LEASE_COUNT" STEPS=4 TP=4 \
    bash "$ROOT/scripts/glm52_linfp8_stacked_coherence.sh" >"$TMP/fail.log" 2>&1; then
  echo "FAIL: missing rank agreement passed" >&2
  exit 1
fi
grep -q 'rank-agreement width changed' "$TMP/fail.log"
[[ "$(wc -l <"$LEASE_COUNT")" -eq 1 ]]

: >"$LEASE_COUNT"
if GLM_COH_SELFTEST_EMPTY_BUILD_CHECKSUM=1 ARMS=stk_base PLOW_AB_DIR="$TMP" \
  PLOW_CKPT_Q="$TMP/checkpoint" PLOW_HSACO="$TMP/objects" \
  PLOWRT_BIN="$TMP/plowrt" PLOW_GPULEASE_BIN="$TMP/gpulease" \
  GLM_COH_LEASE_COUNT="$LEASE_COUNT" STEPS=4 TP=4 \
    bash "$ROOT/scripts/glm52_linfp8_stacked_coherence.sh" >"$TMP/fail-checksum.log" 2>&1; then
  echo "FAIL: empty build-manifest checksum passed" >&2
  exit 1
fi
grep -q 'build-manifest checksum is missing' "$TMP/fail-checksum.log"

mv "$TMP/stk_base.build.json" "$TMP/stk_base.build.json.absent"
: >"$LEASE_COUNT"
if ARMS=stk_base PLOW_AB_DIR="$TMP" PLOW_CKPT_Q="$TMP/checkpoint" \
  PLOW_HSACO="$TMP/objects" PLOWRT_BIN="$TMP/plowrt" \
  PLOW_GPULEASE_BIN="$TMP/gpulease" GLM_COH_LEASE_COUNT="$LEASE_COUNT" STEPS=4 TP=4 \
    bash "$ROOT/scripts/glm52_linfp8_stacked_coherence.sh" >"$TMP/fail-manifest.log" 2>&1; then
  echo "FAIL: missing per-arm build manifest passed" >&2
  exit 1
fi
grep -q 'missing per-arm build manifest' "$TMP/fail-manifest.log"
echo "PASS: stacked coherence requires per-arm manifests, one lease, and every-token TP agreement"
