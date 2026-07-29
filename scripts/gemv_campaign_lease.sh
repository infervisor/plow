#!/usr/bin/env bash
# One GPU, one lease: the decode-GEMV row campaign. See scripts/rebench_tune_gemv.sh's header.
#
# The SWEEP half must run OUTSIDE nix (system ROCm) and UNDER a lease (it is a timing run); the
# INGEST half needs cargo and therefore nix, and must not run under the lease. Both halves are
# here because a missing ingest leaves the store untouched with every gate green.
#
#   $1 object dir (default build-amd/gemvsweep-objs)   $2 jsonl (default /tmp/gemv_sweep.jsonl)
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:-/home/lava/plow/build-amd/gemvsweep-objs}"
JSONL="${2:-/tmp/gemv_sweep.jsonl}"

env GPU_LEASE_TIMEOUT=7200 "$WT/perf-data/harness/gpulease" -n 1 gemv-campaign \
  sg render -c "bash $WT/scripts/rebench_tune_gemv.sh $OBJ $JSONL"
rc=$?
[ "$rc" -eq 0 ] || { echo "sweep failed rc=$rc — NOT ingesting"; exit "$rc"; }

cd "$WT"
nix develop -c cargo run --release -p tunedb --bin tunedb-gemv -- \
    ingest --db tuning --samples "$JSONL" --campaign gemv-row-inventory
