#!/usr/bin/env bash
# sanitize.sh <assets> [extra plowrt bench args] — memcheck one short request.
A=$1; shift
exec $CUDA_PATH/bin/compute-sanitizer --tool memcheck --show-backtrace device --print-limit 5 \
  /root/tts-work/plowrt-probe1 bench --assets "$A" --random-input-len 16 --requests 1 \
  --warmup-requests 0 "$@"
