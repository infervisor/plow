#!/usr/bin/env bash
# Lease + nix wrapper for scripts/l2_place_ab.sh. Keeps the nix shell rooted at the repo (the
# flake lives there) while the A/B itself runs from wherever.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
exec sg render -c "unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; nix develop '$REPO' --command '$REPO/scripts/l2_place_ab.sh'"
