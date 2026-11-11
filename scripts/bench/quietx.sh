#!/usr/bin/env bash
# Exclusive hold of the CPU-quiet lock for a measurement session, writer-preferring: FILE.gate is
# held while waiting for FILE, so builds (quiets.sh) arriving in the meantime queue behind this
# session instead of starving it (flock alone grants new shared holders past a waiting exclusive
# one). Run INSIDE the GPU lease: lease first, then lock.   usage: quietx.sh FILE cmd...
# --check-compilers checks non-cooperating compilers without taking another lock.
set -euo pipefail
check_compilers() {
    local compilers
    compilers=$(ps -eo pid,ppid,stat,comm | awk '
        $3 !~ /^Z/ && $4 ~ /^(rustc|nvcc|cicc|ptxas|cc1|cc1plus|clang|clang\+\+|rust-lld|ld\.lld|lto1)$/')
    if [ -n "$compilers" ]; then
        printf 'Active compiler outside the CPU quiet protocol; measurement is unqualified:\n%s\n' "$compilers" >&2
        return 75
    fi
}
if [ "${1:-}" = --check-compilers ]; then
    check_compilers
    exit
fi
f=$1; shift
exec 8>"$f.gate"; flock -x 8
exec 9>"$f"; flock -x 9
flock -u 8; exec 8>&-
check_compilers
exec "$@"
