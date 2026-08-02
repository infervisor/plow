#!/usr/bin/env bash
# Pull the headline rows out of every `vllm bench serve` log matching a prefix.
for f in "$@"; do
  [ -f "$f" ] || continue
  echo "=== $f"
  grep -E "Successful requests|Maximum request concurrency|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Mean TPOT|Mean ITL" "$f" \
    | sed 's/^/  /'
done
