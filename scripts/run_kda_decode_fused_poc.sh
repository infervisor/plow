#!/usr/bin/env bash
# Runtime half of the standalone KDA fusion gate. gpulease rc=76 remains a failed/contended run.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-${PLOW_BUILD_DIR:-/tmp/plow-kda-decode-fused-poc}}
test -x "$OUT/kda_decode_fused_poc" || { echo "build first: scripts/build_kda_decode_fused_poc.sh $OUT" >&2; exit 2; }
test -f "$OUT/kda_decode_fused_poc.co" || { echo "missing $OUT/kda_decode_fused_poc.co" >&2; exit 2; }
exec "$ROOT/perf-data/tools/gpulease" -n 1 kda-decode-fused-gate \
    "$OUT/kda_decode_fused_poc" "$OUT/kda_decode_fused_poc.co"
