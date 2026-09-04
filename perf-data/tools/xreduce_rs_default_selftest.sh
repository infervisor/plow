#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d /tmp/plow-xreduce-rs-default.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

args=(-DPLOW_GFX950_HSACO=ON -DPLOW_HSACO_FP8=OFF -DPLOW_HSACO_FP8KV=OFF
      -DPLOW_HSACO_MXFP4=OFF -DPLOW_HSACO_MLA=OFF -DPLOW_HSACO_K3=OFF)
cmake -S "$ROOT/runtime" -B "$TMP/gfx950" "${args[@]}" -DPLOW_HSACO_ARCH=gfx950 >/dev/null
cmake -S "$ROOT/runtime" -B "$TMP/gfx942" "${args[@]}" -DPLOW_HSACO_ARCH=gfx942 >/dev/null
cmake -S "$ROOT/runtime" -B "$TMP/rollback" "${args[@]}" -DPLOW_HSACO_ARCH=gfx950 \
  -DPLOW_XR_RS_U=1 >/dev/null

python3 - "$TMP" <<'PY'
import pathlib, sys

def value(root, key):
    prefix = key + ":"
    for line in (pathlib.Path(root) / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise AssertionError(f"missing {key}")

def commands(root):
    lines = []
    for name in ("build.make", "build.ninja"):
        for path in pathlib.Path(root).rglob(name):
            lines.extend(line for line in path.read_text().splitlines()
                         if "interp_prefill" in line and "PLOW_XR_RS_U" in line)
    return lines

root = pathlib.Path(sys.argv[1])
assert value(root / "gfx950", "PLOW_XR_RS_U") == "2"
assert value(root / "gfx942", "PLOW_XR_RS_U") == "1"
assert value(root / "rollback", "PLOW_XR_RS_U") == "1"
assert commands(root / "gfx950")
assert all("-DPLOW_XR_RS_U=2" in line for line in commands(root / "gfx950"))
assert not commands(root / "gfx942")
assert not commands(root / "rollback")
PY

if cmake -S "$ROOT/runtime" -B "$TMP/invalid" "${args[@]}" \
    -DPLOW_HSACO_ARCH=gfx950 -DPLOW_XR_RS_U=4 >/dev/null 2>&1; then
  echo "FAIL: accepted invalid PLOW_XR_RS_U" >&2
  exit 1
fi

echo "xreduce RS default selftest: PASS"
