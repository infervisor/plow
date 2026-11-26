#!/usr/bin/env bash
# until.sh <job id> <seconds>: return when the job leaves `queued`/`running` (or its state changes), or after <seconds>.
F=/tmp/plow-gpuq/$1.json
s0=$(python3 -c "import json;print(json.load(open('$F'))['state'])")
end=$((SECONDS + $2))
while [ $SECONDS -lt $end ]; do
  s=$(python3 -c "import json;print(json.load(open('$F'))['state'])")
  [ "$s" != "$s0" ] && break
  sleep 15
done
echo "state=$s"; tail -5 /home/ifv/plow-scratch/libattn/job.log 2>/dev/null
