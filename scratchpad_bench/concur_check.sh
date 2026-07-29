#!/usr/bin/env bash
# Fire N concurrent chat requests at a live plowrt endpoint and print each
# answer. Two identical questions must give identical text.
PORT="${PORT:-8123}"
SLUG="${SLUG:-842da3794eaa0b77d5f08bae87a17459d91ff475}"
ask() {
  local out
  out=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SLUG\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":32,\"temperature\":0}")
  echo "[$1] $(python3 -c "import json,sys; print(repr(json.loads(sys.argv[1])['choices'][0]['message']['content']))" "$out" 2>/dev/null || echo "PARSE-FAIL: $out")"
}
ask A "What is the capital of France? Answer in one short sentence." &
ask B "What is the capital of France? Answer in one short sentence." &
ask C "What is 2+2? Answer with just the number." &
ask D "Name the largest planet in the solar system in one word." &
ask E "What is the capital of Japan? Answer in one short sentence." &
ask F "What is 10 times 7? Answer with just the number." &
wait
