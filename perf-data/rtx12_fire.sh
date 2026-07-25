#!/usr/bin/env bash
# fire N concurrent chat requests at the live smoke server (R>1 check)
set -u
PORT=8097
for i in 1 2 3 4 5 6 7 8; do
  curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma-4-12b-it\",\"messages\":[{\"role\":\"user\",\"content\":\"Write about the number $i and its prime factorization in detail across several sentences.\"}],\"max_tokens\":60,\"temperature\":0}" >/dev/null &
done
wait
echo "done firing 8 concurrent"
