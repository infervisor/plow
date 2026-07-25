#!/usr/bin/env bash
# Wait until GPU VRAM is released (< 1000 MiB used), then report.
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do
  sleep 5
done
echo "VRAM released: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
