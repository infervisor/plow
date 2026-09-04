#!/usr/bin/env bash
# Production-path TP coherence gate for the stacked GLM_LINEAR_FP8 arm.
#
# Greedy streams legitimately differ between bf16 and fp8. This gate therefore
# does not compare arms: it requires every rank within each TP4 arm to agree on
# every generated token, with the per-dispatch counter audit enabled. Reports
# retain the exact prompt/output and packet/object/checkpoint identities.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${1:-${PLOW_AB_DIR:-/tmp/glmlfp8_stk}}"
CKPT="${PLOW_CKPT_Q:-/home/lava/models/GLM-5.2-plow-q}"
OBJS="${PLOW_HSACO:-$REPO/build-amd/lfp8-stk-objs}"
RT="${PLOWRT_BIN:-$REPO/target/release/plowrt}"
LEASE="${PLOW_GPULEASE_BIN:-$REPO/perf-data/tools/gpulease}"
TP="${TP:-4}"
STEPS="${STEPS:-48}"
IDS="$(cat "${PROMPT_IDS:-$D/prompt_ids.txt}")"
ARMS="${ARMS:-stk_base stk_lfp8}"

if [[ -z "${PLOW_GLM52_COH_LEASED:-}" ]]; then
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec env PLOW_GLM52_COH_LEASED=1 "$LEASE" -n "$TP" glm52-stacked-coherence "$0" "$@"
fi

mkdir -p "$D/coh"
ASSET_ROOT="$(mktemp -d "$D/coh/assets.XXXXXX")"
trap 'rm -rf "$ASSET_ROOT"' EXIT

for arm in $ARMS; do
  packet="$D/$arm.pkt"
  manifest="$D/$arm.build.json"
  assets="$ASSET_ROOT/$arm"
  report="$D/coh/$arm.json"
  log="$D/coh/$arm.log"
  ids="$D/coh/$arm.ids"
  test -s "$manifest" || { echo "FAIL: missing per-arm build manifest: $manifest" >&2; exit 1; }
  mkdir -p "$assets"
  ln -s "$(readlink -f "$packet")" "$assets/model.pkt"
  ln -s "$(readlink -f "$manifest")" "$assets/build.json"
  cat >"$assets/weights.json" <<JSON
{
  "network": "glm-5.2-$arm",
  "gpu": "mi355x",
  "num_gpus": $TP,
  "parallel": "tp",
  "weight_shared": false,
  "weight": null,
  "kv": null,
  "fusion": null,
  "buckets": [],
  "static_tensors": [],
  "static_tensors_file_emitted": false,
  "weight_tiling": null
}
JSON

  echo "--- $arm (production TP agreement) ---"
  env -u PLOW_HSACO_LOWRUNG \
    "$RT" --rt-checkpoint "$CKPT" --rt-hsaco "$OBJS" \
      --amd-tp-agree-every 1 --amd-tp-no-audit=false \
      bench --assets "$assets" --prompt-ids "$IDS" \
      --concurrency 1 --requests 1 --warmup-requests 0 \
      --output-len "$STEPS" --token-audit --engine-diagnostics \
      >"$report" 2>"$log" || { cat "$log" >&2; exit 1; }

  python3 - "$report" "$ids" "$arm" "$assets" "$packet" "$manifest" "$OBJS" \
    "$CKPT" "$IDS" "$STEPS" "$TP" <<'PY'
import json, os, sys

(report_path, ids_path, arm, assets, packet, manifest, objects, checkpoint,
 prompt_raw, steps, tp) = sys.argv[1:]
steps, tp = int(steps), int(tp)
prompt = [int(token.strip()) for token in prompt_raw.split(",")]
with open(report_path) as source:
    report = json.load(source)

def need(ok, message):
    if not ok:
        raise SystemExit(f"{arm}: {message}")

need(report.get("schema") == "plowrt.bench.v1", "unexpected bench schema")
need(report.get("vendor") == "Some(Amd)", "production bench did not use AMD")
need(report.get("num_gpus") == tp and report.get("parallel") == "tp",
     "production engine did not load the requested TP width")
need((report.get("requests"), report.get("completed"), report.get("failed")) == (1, 1, 0),
     "measured request did not complete exactly once")
need(report.get("warmup_requests") == 0, "unexpected warmup request")
need(report.get("prompt_tokens") == len(prompt), "prompt token count changed")
need(report.get("output_tokens") == steps, "output token count changed")
schedule = report.get("scheduler") or {}
need(schedule.get("rejected") == 0 and schedule.get("admit_shed") == 0,
     "production scheduler rejected or shed work")

audit = report.get("token_audit") or {}
rows = audit.get("output_token_ids")
need(audit.get("prompt_token_ids") == [prompt], "token audit prompt changed")
need(isinstance(rows, list) and len(rows) == 1 and len(rows[0]) == steps,
     "token audit returned an incomplete stream")
need(any(rows[0]), "all-zero output makes coherence vacuous")

real = os.path.realpath
artifacts = report.get("artifacts") or {}
need(real(report.get("asset_dir", "")) == real(assets), "bench loaded a different asset directory")
need(real((artifacts.get("packet") or {}).get("path", "")) == real(packet),
     "bench loaded a different packet")
build = artifacts.get("build_manifest") or {}
need(bool(build.get("checksum")), "build-manifest checksum is missing")
need(real(build.get("path", "")) == real(manifest),
     "bench loaded a different build manifest")
need(real((artifacts.get("checkpoint") or {}).get("path", "")) == real(checkpoint),
     "bench bound a different checkpoint")
inventory = artifacts.get("object_inventory") or []
object_root = real(objects)
need(inventory and all(os.path.commonpath([real(item.get("path", "")), object_root]) == object_root
                       for item in inventory),
     "bench inventoried objects outside the selected directory")

diagnostics = report.get("diagnostics") or {}
agreement = diagnostics.get("rank_agreement") or {}
need(diagnostics.get("supported") is True and diagnostics.get("complete") is True and
     diagnostics.get("overflowed") is False, "missing or partial engine diagnostics")
need(agreement.get("ranks") == tp, "rank-agreement width changed")
need(agreement.get("sampled_token_every") == 1, "not every token was checked across ranks")
need(agreement.get("counter_audit_every_dispatch") is True,
     "per-dispatch TP counter audit was disabled")
need(agreement.get("prefill_completion_all_ranks") is True,
     "prefill completion did not cover every rank")

with open(ids_path, "w") as output:
    output.write(",".join(map(str, rows[0])) + "\n")
print(f"[{arm}] TP{tp} EVERY-TOKEN AGREEMENT; {steps} complete output tokens; report {report_path}")
PY
done

echo "===== generated ids (detokenize with the model tokenizer to read them) ====="
for arm in $ARMS; do printf '%s: ' "$arm"; cat "$D/coh/$arm.ids"; done
