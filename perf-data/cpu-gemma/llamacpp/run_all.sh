#!/usr/bin/env bash
# Serialized driver: wait for the Gemma Paris smoke test, then bench every model, then gpt-oss sanity.
set -uo pipefail
D=/home/lava/llamacpp
while pgrep -f "$D/paris.sh" >/dev/null; do sleep 15; done
for n in bf16 q8_0 q4_k_m gptoss; do
  echo "=== run_bench $n $(date +%T)"
  $D/run_bench.sh $n
  echo "=== rc=$? $(date +%T)"
done
$D/paris-gptoss.sh
echo "=== ALL DONE $(date +%T)"
