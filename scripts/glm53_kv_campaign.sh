#!/usr/bin/env bash
# The whole decode-KV campaign under ONE 4-GPU lease.
#
# Every arm here is a separate `plowrt serve`, and taking a lease per arm means
# queueing behind every other agent's single-GPU job between arms — on a busy box
# that is most of the wall clock. So the lease is acquired once, around the whole
# sequence, and the arms run back to back inside it.
#
#   ./scripts/glm53_kv_campaign.sh            # takes the lease, runs everything
#   ARMS="ctl fp8" ./scripts/glm53_kv_campaign.sh   # a subset
#
# The inner half (`glm53_kv_campaign_inner.sh`) does the serving; it only ever
# signals plowrt PIDs it started itself.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-14400}" PLOW_TP_NO_AUDIT=1 \
  /app/plow/perf-data/tools/gpulease -n 4 "glm53-decode-kv" \
  nix develop /app/plow --command bash "$WT/scripts/glm53_kv_campaign_inner.sh"
