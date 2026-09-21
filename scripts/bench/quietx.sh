#!/usr/bin/env bash
# Exclusive hold of the CPU-quiet lock for a measurement session, writer-preferring: FILE.gate is
# held while waiting for FILE, so builds (quiets.sh) arriving in the meantime queue behind this
# session instead of starving it (flock alone grants new shared holders past a waiting exclusive
# one). Run INSIDE the GPU lease: lease first, then lock.   usage: quietx.sh FILE cmd...
set -euo pipefail
f=$1; shift
exec 8>"$f.gate"; flock -x 8
exec 9>"$f"; flock -x 9
flock -u 8; exec 8>&-
exec "$@"
