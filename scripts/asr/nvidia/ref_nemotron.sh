#!/usr/bin/env bash
# Nemotron 3.5 ASR transformers baselines (offline + cache-aware streaming). Run under gpulease.
set -u
PY=${PY:-/root/asr-work/venv/bin/python}
MODEL=${MODEL:-/root/plow/models/nemotron-3.5-asr-streaming-0.6b}
OUT=${OUT:-/root/asr-work/results}
HERE=$(dirname "$0")
"$PY" "$HERE/ref_bench.py" nemotron --model "$MODEL" --out "$OUT/nemo_offline.json" --profile x > "$OUT/nemo_offline.log" 2>&1
echo "offline rc=$?"
"$PY" "$HERE/ref_bench.py" nemotron --model "$MODEL" --stream --out "$OUT/nemo_stream.json" --profile x > "$OUT/nemo_stream.log" 2>&1
echo "stream rc=$?"
