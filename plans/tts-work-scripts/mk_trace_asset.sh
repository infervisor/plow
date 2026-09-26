#!/usr/bin/env bash
# mk_trace_asset.sh <src-assets> <dst> [extra -D define...] — asset copy whose objects are rebuilt
# from the source asset's CMake configuration plus extra defines (default -DPLOW_NV_TRACE=1).
set -e
source /root/tts-work/cuda-env.sh
SRC=$1; DST=$2; shift 2
ADD="${*:--DPLOW_NV_TRACE=1}"
rm -rf "$DST"; mkdir -p "$DST"
for f in "$SRC"/*; do ln -s "$(readlink -f "$f")" "$DST/$(basename "$f")"; done
# Every build output must be a fresh file here, never a link into the source asset.
rm -f "$DST"/*.cubin "$DST/codec"
mkdir -p "$DST/codec" && ln -s "$(readlink -f "$SRC/codec/snac24k.bin")" "$DST/codec/snac24k.bin"
ARGS=()
while IFS= read -r line; do
  k=${line%%:*}; v=${line#*=}
  case "$k" in
    PLOW_CUBIN_DIR) ;;
    PLOW_EXTRA_DEFINES) ARGS+=("-D$k=$v $ADD") ;;
    *) ARGS+=("-D$k=$v") ;;
  esac
done < <(grep -E '^PLOW_[A-Z0-9_]+:(STRING|BOOL|FILEPATH|PATH)=' "$SRC/.cubin-build/CMakeCache.txt")
cmake -S /root/plow/.claude/worktrees/tts-veena-chatterbox/runtime -B "$DST/.cubin-build" \
  "${ARGS[@]}" -DPLOW_CUBIN_DIR="$DST" > "$DST/cmake.log" 2>&1 || { tail -n 20 "$DST/cmake.log"; exit 1; }
cmake --build "$DST/.cubin-build" --target sm120_cubins > "$DST/build.log" 2>&1 || { grep -i error "$DST/build.log" | head; exit 1; }
grep "^built" "$DST/build.log"
