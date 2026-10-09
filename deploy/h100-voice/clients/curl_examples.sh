#!/usr/bin/env bash
# curl against every HTTP endpoint of the voice stack. PLOW_URL (default http://127.0.0.1:8000) and
# PLOW_API_KEY (optional) as for the Python clients. Writes its outputs to ${OUT:-./curl-out}.
#   bash clients/curl_examples.sh [clients/samples/sample_en.wav]
set -euo pipefail
URL=${PLOW_URL:-http://127.0.0.1:8000}
WAV=${1:-$(dirname "$0")/samples/sample_en.wav}
OUT=${OUT:-./curl-out}; mkdir -p "$OUT"
AUTH=(); [ -n "${PLOW_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $PLOW_API_KEY")
set -x

# Health (no key needed), Prometheus metrics, model list
curl -fsS "$URL/health"; echo
curl -fsS "${AUTH[@]}" "$URL/metrics" | grep -c '^plow' || true
curl -fsS "${AUTH[@]}" "$URL/v1/models" | python3 -m json.tool | grep '"id"'

# Transcription: JSON, plain text, and streamed (server-sent events)
curl -fsS "${AUTH[@]}" "$URL/v1/audio/transcriptions" -F model=qwen3-asr -F language=en -F file=@"$WAV"; echo
curl -fsS "${AUTH[@]}" "$URL/v1/audio/transcriptions" -F model=qwen3-asr -F response_format=text -F file=@"$WAV"; echo
curl -fsSN "${AUTH[@]}" "$URL/v1/audio/transcriptions" -F model=qwen3-asr -F stream=true -F file=@"$WAV"

# Voice activity (Silero VAD, when the profile loads it): speech segments in seconds
curl -fsS "${AUTH[@]}" "$URL/v1/audio/vad" -F file=@"$WAV"; echo

# Speech: complete WAV, and streamed raw PCM (s16le mono 24 kHz)
curl -fsS "${AUTH[@]}" "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"chatterbox-mtl","input":"Hello from the voice stack.","voice":"default","response_format":"wav"}' \
  -o "$OUT/speech.wav"
curl -fsSN "${AUTH[@]}" "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"chatterbox-mtl","input":"Bonjour tout le monde.","voice":"default","language":"fr","response_format":"pcm","stream":true}' \
  -o "$OUT/speech_fr.pcm"
ls -l "$OUT/speech.wav" "$OUT/speech_fr.pcm"

# Chat (non-streamed, streamed with logprobs) and raw completion (the prompt must start with <bos>)
curl -fsS "${AUTH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'; echo
curl -fsSN "${AUTH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Count to three."}],"max_tokens":16,"temperature":0,"stream":true,"logprobs":true,"top_logprobs":2}' \
  -o "$OUT/chat_stream.sse"
head -c 600 "$OUT/chat_stream.sse"; echo
curl -fsS "${AUTH[@]}" "$URL/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b","prompt":"<bos>The capital of France is","max_tokens":8,"temperature":0}'; echo
