#!/usr/bin/env bash
# Run through gpuq with a frozen packet, objects, runtime, prompt and manifest tool.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/plowbench.sh"
pb_require_nix
pb_hazard_env
capture=$(realpath "${1:?frozen capture directory required}")
checkpoint=$(realpath "${2:?prepared checkpoint required}")
batch=${3:?batch size required}
steps=${4:-4}
tp=${5:-8}
vocab=${6:-154880}
mode_name=${7:-capture}
[[ "$mode_name" == capture || "$mode_name" == trace ]] || { echo 'mode must be capture or trace' >&2; exit 2; }
for value in "$batch" "$steps" "$tp" "$vocab"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'dimensions must be positive integers' >&2; exit 2; }
done
(( tp > 1 )) || { echo 'TP must exceed one' >&2; exit 2; }
cd "$capture"
sha256sum --check SHA256SUMS
mkdir outputs
export PLOW_TP_NO_AUDIT=0 PLOW_TP_AGREE_EVERY=1 PLOW_PREFIX_CACHE=0
export PLOW_DUMP_ACT="act.logits:$capture/outputs/logits:$((2 * vocab * batch))"
if [[ "$mode_name" == trace ]]; then
    export PLOW_TRACE_RAW="$capture/outputs/trace.bin"
fi
mode=()
if (( batch > 1 )); then mode=(--batched); fi
"$capture/plowrt" amd-bench --blob "$capture/assets/model.pkt" \
    --hsaco "$capture/objects" --checkpoint "$checkpoint" \
    --prompt "@$capture/prompt.txt" --tp "$tp" --steps "$steps" "${mode[@]}" \
    2>&1 | tee "$capture/outputs/stdout.log"
python3 "$capture/plow_logit_manifest.py" --name "c$batch" \
    --prompt "$capture/prompt.txt" --stdout "$capture/outputs/stdout.log" \
    --logits-dir "$capture/outputs" --output "$capture/outputs/manifest.json" \
    --tp-shards "$tp" --vocab "$vocab" --batch-size "$batch" --replicated-vocab
