#!/usr/bin/env bash
# Wait for a vLLM container to become healthy. $1 port, $2 container name, $3 max 10s ticks.
set -u
P="${1:?port}"; C="${2:?name}"; N="${3:-360}"
for i in $(seq 1 "$N"); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$P/health")" = "200" ]; then
    echo "HEALTHY after ~$((i * 10))s"; exit 0
  fi
  if ! sudo -n docker ps --format '{{.Names}}' | grep -q "^${C}$"; then
    echo "EXITED after ~$((i * 10))s"; exit 2
  fi
  sleep 10
done
echo "TIMEOUT"; exit 1
