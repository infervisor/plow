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
# A GPU memory fault makes ROCr dump EVERY resident GPU segment to a file. On this box that is
# ~160 GB written to the root overlay, which takes ~20 minutes: the server looks wedged (state S,
# no TICK lines, in-flight requests hanging) and the disk other campaigns share fills up.
# Measured: gpucore.2119.gpu, 160,324,931,952 bytes, from one fault at 70k/C16.
# So the fault aborts promptly by default. Set HSA_DISABLE_COREDUMP_ON_EXCEPTION=0 to collect one
# deliberately, and prefer HSA_ENABLE_LIGHTWEIGHT_COREDUMP=1 plus an HSA_COREDUMP_PATTERN under
# /workspace -- never the default path, which lands in the repo.
export HSA_DISABLE_COREDUMP_ON_EXCEPTION="${HSA_DISABLE_COREDUMP_ON_EXCEPTION:-1}"
export PLOW_MLA_PF_V2=1
# GLM53_RECIPE carries BOTH `env.PLOW_MLA_PF_V2` and `env.PLOW_MLA_PF_AITER`; only the first was
# ever set here. `rt.mla_pf_aiter` is OptIn/OFF, and it alone decides whether the AITER sparse-MLA
# route loads (exec/amd.rs:6882). Without it a packed/token-batch program carrying the sparse
# chain is REFUSED at route time -- "the sparse FP8 flash needs the AITER route ... the packed
# flash object has no gathered arm" -- and the whole sparse prefill silently falls back to the
# unpacked path. Observed 9x in /workspace/dsapf-full/server.log. The message names an object arm,
# but the test is `RuntimeConfig::get().amd.mla_pf_aiter`, which is this variable.
export PLOW_MLA_PF_AITER=1
# NOT REQUIRED ON AMD, and kept only so an operator copying this line onto the CUDA path gets
# the same behaviour. `AmdEngine::load` accepts an L2-placed blob unconditionally and then
# checks each code object for `plow_l2_place_dispatch_1`, which is a stronger guard than this
# assertion: verified by serving `PLOWDEV\x0b` with the variable unset and reading
# "L2 hierarchical gate: FIRING" out of the log.
export PLOW_L2_PLACE_DISPATCH=1
# The runtime discovers tiers matching the packet precision and scheduler.
exec "$BIN/plowrt" serve --assets "$ASSETS" --port "$PORT"
