#!/usr/bin/env bash
# wait.sh <job id>: block until the gpuq job reaches a terminal state; print it.
F=/tmp/plow-gpuq/$1.json
while true; do
  s=$(python3 -c "import json;print(json.load(open('$F'))['state'])" 2>/dev/null)
  case "$s" in done|failed|cancelled|error|killed|timeout|lost) echo "state=$s"; python3 -c "import json;d=json.load(open('$F'));print(d.get('rc'))"; exit 0 ;; esac
  sleep 30
done
