#!/usr/bin/env bash
# Needle retrieval for the bf16/fp8 latent pair, under ONE 4-GPU lease.
# Split from `glm53_kv_campaign.sh` because it was written after that campaign
# had already taken its lease, and a script must not be edited while a running
# job is reading it.
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env GPU_LEASE_TIMEOUT="${GPU_LEASE_TIMEOUT:-14400}" PLOW_TP_NO_AUDIT=1 \
  /app/plow/perf-data/tools/gpulease -n 4 "glm53-kv-needle" \
  nix develop /app/plow --command bash "$WT/scripts/glm53_needle_campaign_inner.sh"
