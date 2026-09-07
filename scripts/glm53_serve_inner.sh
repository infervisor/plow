#!/usr/bin/env bash
# The INSIDE-nix half of `glm53_mi300x.sh serve`. Split out so no quoting has to survive
# gpulease -> nix develop -> bash -c.
#
# WHY INSIDE NIX AT ALL: libhsa-runtime64.so.1 lives in the flake's ROCm 7.14 tree, which is
# also the toolchain build_gfx942.sh compiled the objects with. Outside it the dlopen fails and
# plowrt falls back to the CPU REFERENCE interpreter — which serves coherent answers at
# fictional speed. That happened on the first attempt here.
set -uo pipefail
ASSETS="${1:?assets}"; PORT="${2:?port}"; OBJ="${3:?objdir}"; BIN="${4:?bindir}"
export LD_LIBRARY_PATH="${ROCM_PATH:?nix develop did not set ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PLOW_HSACO="$OBJ"
export PLOW_MLA_PF_V2=1
export PLOW_L2_PLACE_DISPATCH=1
# Optional narrower decode tier, e.g. PLOW_HSACO_LOWRUNG=<dir>:4 to serve rungs 1..4 from a
# register-squeezed object. Echoed because build.json cannot record it and two runs that differ
# only in this knob are otherwise indistinguishable in the log.
echo "serve tiers: PLOW_HSACO=$PLOW_HSACO PLOW_HSACO_LOWRUNG=${PLOW_HSACO_LOWRUNG:-<none>}"
exec "$BIN/plowrt" serve --assets "$ASSETS" --port "$PORT"
