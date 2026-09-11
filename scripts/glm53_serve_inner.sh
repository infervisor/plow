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
# NOT REQUIRED ON AMD, and kept only so an operator copying this line onto the CUDA path gets
# the same behaviour. `AmdEngine::load` accepts an L2-placed blob unconditionally and then
# checks each code object for `plow_l2_place_dispatch_1`, which is a stronger guard than this
# assertion: verified by serving `PLOWDEV\x0b` with the variable unset and reading
# "L2 hierarchical gate: FIRING" out of the log.
export PLOW_L2_PLACE_DISPATCH=1
# The runtime discovers tiers matching the packet precision and scheduler.
exec "$BIN/plowrt" serve --assets "$ASSETS" --port "$PORT"
