#!/usr/bin/env bash
# The GEMV DEMAND CENSUS: read the compiler's own question list back out of it.
#
# `PLOW_TUNE_DUMP=1` makes `packet::devbuild::Builder::emit_dep` print one `TUNEDUMP_GEMV` line
# per emitted decode-GEMV op. This runs an emit and reduces that to the distinct
# `(M, N, K, quant, opcode)` set, which IS the campaign list for
# `scripts/rebench_tune_gemv.sh` — derived, not authored.
#
# Hand-authoring is not a style preference here. `scripts/rebench_tune_gemm.sh`'s list WAS
# hand-authored, and GLM-5.2 prefill was consequently 100% unmeasured for the tuner's entire
# life while every gate stayed green. Re-run this after ANY emitter change.
#
#   $1  model: glm | gemma31b   (default glm)
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-glm}"
LOG="${LOG:-/tmp/gemv_census_$MODEL.log}"
cd "$WT"

case "$MODEL" in
  glm)
    PLOW_TUNE_DUMP=1 GLM_CTX="${GLM_CTX:-32768}" \
      nix develop -c bash "$WT/scripts/rebench_emit_glm.sh" "${OUT:-/tmp/gemv_census.pkt}" \
      >"$LOG" 2>&1
    ;;
  gemma31b)
    CKPT="${GEMMA31B:-/home/lava/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475}"
    PLOW_TUNE_DUMP=1 PLOW_DECODE_BATCH="${PLOW_DECODE_BATCH:-16}" \
    PLOW_GEMV_MM="${PLOW_GEMV_MM:-8}" PLOW_GEMV_WALK="${PLOW_GEMV_WALK:-1}" \
      nix develop -c "$WT/target/release/plowc" --hf-dir "$CKPT" --emit devblob \
        --max-ctx "${MAXCTX:-4096}" --n-cu 256 --out "${OUT:-/tmp/gemv_census.pkt}" \
      >"$LOG" 2>&1
    ;;
  *) echo "usage: $0 glm|gemma31b" >&2; exit 2;;
esac
rc=$?

echo "== GEMM census (the tuned path)"
grep '^TUNEDUMP ' "$LOG" | awk '{print $NF}' | sort | uniq -c
echo
echo "== GEMV census (this is the campaign list)"
grep -c TUNEDUMP_GEMV "$LOG" | sed 's/^/  emits: /'
grep TUNEDUMP_GEMV "$LOG" | awk '{print $NF}' | sort | uniq -c | sed 's/^/  /'
echo "  distinct shapes:"
# `sort -u` over the WHOLE line, then sort for display. `sort -u -k2,2n` would dedupe on the
# key ALONE and silently drop every shape sharing an N — it hid three of Gemma-31B's seven
# down-projection shapes on the first run, which would have produced an under-covered campaign
# list from a script whose entire job is to stop exactly that.
grep TUNEDUMP_GEMV "$LOG" | awk '{printf "    %8s %8s %8s  %-6s %s\n", $2, $3, $4, $5, $6}' \
  | sort -u | sort -k2,2n -k3,3n
exit $rc
