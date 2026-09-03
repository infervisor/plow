#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d /tmp/plow-inventory-default.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP/plow_config.h" <<'PY'
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    "#define PLOW_PACKET_HAS_KDA_DECODE_FUSED 1\n"
)
PY

args=(
  -DPLOW_GFX950_HSACO=ON
  -DPLOW_HSACO_FP8=OFF
  -DPLOW_HSACO_FP8KV=OFF
  -DPLOW_HSACO_MXFP4=OFF
  -DPLOW_HSACO_MLA=OFF
  -DPLOW_HSACO_K3=OFF
)
cmake -S "$ROOT/runtime" -B "$TMP/no-config" "${args[@]}" >/dev/null
cmake -S "$ROOT/runtime" -B "$TMP/paired" "${args[@]}" \
  -DPLOW_HSACO_CONFIG="$TMP/plow_config.h" >/dev/null
cmake -S "$ROOT/runtime" -B "$TMP/rollback" "${args[@]}" \
  -DPLOW_HSACO_CONFIG="$TMP/plow_config.h" \
  -DPLOW_HSACO_DECODE_INVENTORY_PRUNE=OFF >/dev/null

python3 - "$TMP/no-config" "$TMP/paired" "$TMP/rollback" "$ROOT" <<'PY'
import pathlib
import sys

def cache(root):
    for line in (pathlib.Path(root) / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("PLOW_HSACO_DECODE_INVENTORY_PRUNE:BOOL="):
            return line.rsplit("=", 1)[1]
    raise AssertionError(f"missing inventory-prune cache entry in {root}")

def commands(root):
    lines = []
    for name in ("build.make", "build.ninja"):
        for path in pathlib.Path(root).rglob(name):
            lines.extend(line for line in path.read_text().splitlines()
                         if "-DPLOW_DECODE_INVENTORY_PRUNE=1" in line)
    return lines

assert cache(sys.argv[1]) == "OFF"
assert cache(sys.argv[2]) == "ON"
assert cache(sys.argv[3]) == "OFF"
enabled = commands(sys.argv[2])
assert enabled
assert all("-DPLOW_BUCKET_DECODE=1" in line for line in enabled), enabled
assert not commands(sys.argv[1])
assert not commands(sys.argv[3])

script = pathlib.Path(sys.argv[4]) / "scripts" / "build_gfx950.sh"
text = script.read_text()
assert 'PLOW_HSACO_DECODE_INVENTORY_PRUNE:-auto' in text
assert 'DEC="$DEC -DPLOW_DECODE_INVENTORY_PRUNE=1"' in text
assert 'PLOW_HSACO_DECODE_INVENTORY_PRUNE requires PLOW_HSACO_CONFIG' in text
PY

echo "decode inventory default selftest: PASS"
