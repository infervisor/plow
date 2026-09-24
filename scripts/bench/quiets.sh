#!/usr/bin/env bash
# Shared hold of the CPU-quiet lock for a build or other CPU-heavy job: pass the gate (blocks while
# a measurement session is waiting for the lock), then hold FILE shared with the descriptor closed
# before exec (-o), so a daemon the job spawns (the sccache server) cannot inherit the lock and keep
# it after the job ends.   usage: quiets.sh FILE cmd...   (typically: quiets.sh FILE nice -n 19 cargo ...)
set -euo pipefail
f=$1; shift
flock -x "$f.gate" -c true
exec flock -s -o "$f" "$@"
