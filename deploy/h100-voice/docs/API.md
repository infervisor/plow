# plow-voice API reference

One HTTP/WebSocket server (`plowrt serve`) hosts every model of the profile. The API follows
OpenAI's where one exists (the `openai` SDKs work with `base_url=http://<host>:<port>/v1`), with
a few documented extensions. Examples use `http://127.0.0.1:8000`; every example here is runnable
from `clients/` (`clients/curl_examples.sh` runs the curl ones).

Models in the default `voice-core` profile:

| id | kind | endpoints |
|---|---|---|
| `qwen3-asr` | speech recognition (Qwen3-ASR 1.7B, multilingual, language auto-detect) | `/v1/audio/transcriptions` (+ SSE), `/v1/audio/transcriptions/stream` (WebSocket), `/v1/realtime` |
| `chatterbox-mtl` | text to speech (Chatterbox Multilingual V3, 23 languages) | `/v1/audio/speech` |
| `gemma-4-e4b` | chat LLM (Gemma-4 E4B-it, BF16, 8192-token context) | `/v1/chat/completions`, `/v1/completions` |
| Silero VAD | voice activity (CPU) | `/v1/audio/vad`; turn detection for the streaming APIs |

Other profiles add `qwen3-asr-0.6b`, `nemotron-3.5-asr` (English streaming RNNT), `veena`
(Indic + English TTS), `chatterbox` (English TTS); `GET /v1/models` lists what is served.

## Authentication

With `API_KEYS` configured, every route except `/health` and `/healthz` needs
`Authorization: Bearer <key>` or `x-api-key: <key>`; otherwise 401. Browser WebSocket clients that
cannot set headers may pass the key as the subprotocol `openai-insecure-api-key.<key>` next to
`realtime` (Realtime API only). There is no TLS in the server: see DEPLOY.md "TLS".

## Errors

Errors use OpenAI's envelope: `{"error": {"message": "...", "type": "...", "code": "..."}}`.

| status | when |
|---|---|
| 400 | malformed request, unknown field or unsupported option, context too long (`context_length_exceeded`), bad audio, a model on a route it does not serve (`model_not_supported_for_endpoint`: e.g. a TTS or ASR model on `/v1/chat/completions`; each model's routes are its `x_plow_endpoints` in `/v1/models`) |
| 401 | API keys configured and none / a wrong one sent |
| 404 | unknown model (`model_not_found`) or route |
| 408 | an upload body did not arrive within 30 s |
| 413 | audio longer than the endpoint's limit, body over the size limit |
| 415 | unsupported WAV encoding or sample rate |
| 429 | overloaded: queue or stream capacity full; `Retry-After` header (seconds). Retry later |
| 500 | engine failure for this request |
| 503 | shutting down (drain), or an engine is unavailable; `/health` is 503 too |
| 504 | a transcription missed its deadline (`ASR_REQUEST_TIMEOUT_MS`, 120 s) |

A streamed response that fails after its 200 header ends with an error object as its last SSE
event (`data: {"error": {...}}`, no `[DONE]`), never as generated text.

## Service endpoints

* `GET /health`, `GET /healthz`: `ok` (200) when every model is loaded; 503 while draining or
  after an engine fault. No auth.
* `GET /metrics`: Prometheus text exposition (per-model request counts, latency histograms, queue
  depth, KV use; `vllm:`-compatible series for the LLM).
* `GET /v1/models`, `GET /v1/models/{id}`: OpenAI model cards. Extensions: `x_plow_endpoints`
  (the endpoints the model serves) and `max_model_len` (the served context).

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

## Speech recognition

Audio input everywhere: 16-bit (or 8/24/32-bit integer, 32-bit float) PCM WAV, mono or stereo
(averaged), 8 to 48 kHz (resampled to 16 kHz on the server), 0.5 to 30 s per utterance.
Longer audio: the WebSocket continuous mode or the Realtime API, which segment it at pauses.

### POST /v1/audio/transcriptions

Multipart form, as OpenAI's:

| field | |
|---|---|
| `file` | WAV, at most 4 MiB |
| `model` | `qwen3-asr` (or another ASR id) |
| `language` | optional, e.g. `en`, `English`, `zh`; omitted = detected |
| `prompt` | optional context text (up to 256 tokens) |
| `response_format` | `json` (default, `{"text": ...}`) or `text` |
| `temperature` | only `0` |
| `stream` | `true`: server-sent events `transcript.text.delta` `{delta}` as text is decoded, then `transcript.text.done` `{text, language}` |

With Silero VAD loaded, an upload with under 250 ms of detected speech returns an empty
transcript (`{"text": ""}`) without running the model.

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions -F model=qwen3-asr -F file=@clients/samples/sample_en.wav
# {"text":"Nor is Mister Quilter's manner less interesting than his matter."}
curl -sN http://127.0.0.1:8000/v1/audio/transcriptions -F model=qwen3-asr -F stream=true -F file=@clients/samples/sample_en.wav
# data: {"delta":"Nor","request_id":"...","type":"transcript.text.delta"}
# ...
# data: {"final":true,"language":"English","text":"Nor is ...","type":"transcript.text.done"}
```

Python: `clients/transcribe.py` (`--stream`), or the `openai` SDK (`clients/openai_sdk.py`):
`client.audio.transcriptions.create(model="qwen3-asr", file=f, stream=True)`.

Concurrency: up to 256 uploads in flight; past that 429 with `Retry-After: 1`.

### GET /v1/audio/transcriptions/stream (WebSocket, native protocol v1)

Streaming microphone audio with credit-based flow control. Example: `clients/transcribe_ws.py`.

1. Client sends JSON `start`:
   `{"type":"start","version":1,"model":"qwen3-asr","sample_rate":16000,"format":"pcm_s16le"}`.
   Optional: `language`, `prompt`, `partials` (bool: revisable partial transcripts while audio
   arrives, about one per second), `deltas` (bool: stream the final transcript as `delta` events),
   `mode` (`utterance`, default, or `continuous`), and for continuous mode `min_silence_ms`
   (200-2000, default 600) and `max_segment_ms` (4000 up to the model limit, default 25000).
   `sample_rate`: 8000, 16000, 22050, 24000, 32000, 44100 or 48000.
2. Server sends `ready`: `session_id`, `request_id`, `sample_rate`, `max_chunk_bytes` (32000),
   `credit_samples` (the initial grant), `max_audio_samples`, `partial_mode`, `deltas`, `mode`.
3. Client sends binary frames: little-endian `u64` sequence number (from 0) followed by mono s16le
   PCM, never more samples than the outstanding credit; the server grants more with
   `{"type":"credit","credit_samples":N}` (up to 1 s at a time).
4. Client sends `{"type":"finish"}` after the last audio (`{"type":"cancel"}` aborts).
5. Server events: `partial {revision, text, language, stable_prefix_bytes}`, `delta {text}`,
   `final {revision, text, language, stable_prefix_bytes, turn_id, server_timing}`; in
   continuous mode every event carries `segment`, each final has `start_ms` / `end_ms`, and the
   session ends with `{"type":"done","segments":N}`. Errors: `error {message, terminal, code?}`
   (`code`: `overloaded`, `unavailable`, `timeout`).

Utterance mode: one utterance (up to 30 s) per connection. Continuous mode: unbounded audio, cut
into segments at pauses (Silero VAD speech probability when loaded, else an energy endpointer;
200 ms of context kept on each side; an overlong segment is cut at its quietest point). Limits:
256 sessions; 30 s without audio or control messages ends a session; the server pings every 15 s
and drops a peer silent for 45 s. On shutdown a live session gets an `error` and close code 1001.

### GET /v1/realtime?intent=transcription (OpenAI Realtime, transcription sessions)

OpenAI's Realtime transcription protocol; the `openai` SDK realtime client and browser code work
unchanged. Example: `clients/realtime_transcribe.py`; turn-detection test:
`perf/realtime_vad_test.py`.

* Connect (`?model=qwen3-asr` optional): server sends `transcription_session.created`.
* `transcription_session.update` (or the GA `session.update` with a `transcription` session)
  sets `input_audio_format` (`pcm16` = 24 kHz mono s16le, `g711_ulaw`, `g711_alaw` = 8 kHz),
  `input_audio_transcription` `{model, language, prompt}`, and `turn_detection`; answered with
  `transcription_session.updated`.
* `input_audio_buffer.append {audio: <base64>}` (at most 1 MiB per event).
* `turn_detection: {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 500}`
  (the default): the server cuts turns itself and sends `input_audio_buffer.speech_started
  {audio_start_ms, item_id}`, `input_audio_buffer.speech_stopped {audio_end_ms, item_id}`,
  `input_audio_buffer.committed {item_id, previous_item_id}`. Semantics with Silero VAD (loaded
  in every ASR profile of this kit): each 32 ms frame gets a Silero speech probability; speech
  starts at `threshold` (0..1, default 0.5) and counts as silence below `threshold - 0.15`; a turn
  ends after `silence_duration_ms` (200..2000, default 500) of silence. `prefix_padding_ms` is
  accepted and echoed but fixed at 200 ms of context. `semantic_vad` is not implemented.
* `turn_detection: null`: audio accumulates (up to 30 s) until `input_audio_buffer.commit`
  (at least 100 ms); `input_audio_buffer.clear` drops it.
* Results per committed turn, in order: `conversation.item.input_audio_transcription.delta
  {item_id, content_index, delta}` then `.completed {item_id, transcript}`, or `.failed
  {error}` (`timeout`, `rate_limit_exceeded`).
* Limits: two turns transcribing and 16 waiting per session; 120 s without client events closes
  the session. Not implemented: responses / conversation items, noise reduction, logprobs, usage.

### POST /v1/audio/vad (Silero VAD)

Speech segments of a recording (the VAD runs on the server CPU). Multipart: `file` (WAV, 8-48
kHz, up to 10 minutes, 32 MiB), optional `threshold` (0.5), `min_speech_duration_ms` (250),
`min_silence_duration_ms` (100), `speech_pad_ms` (30), `max_speech_duration_s` (unbounded): the
parameters and defaults of Silero's `get_speech_timestamps`.

```bash
curl -s http://127.0.0.1:8000/v1/audio/vad -F file=@data/librispeech-dummy/ls02.wav
# {"duration":12.545,"speech_duration":12.39,"segments":[{"start":0.034,"end":12.45}]}
```

## Text to speech: POST /v1/audio/speech

JSON body (OpenAI's, plus `language` and `seed`):

| field | |
|---|---|
| `model` | `chatterbox-mtl` (or `veena`, `chatterbox`) |
| `input` | text, up to 4096 characters |
| `voice` | `chatterbox-mtl` / `chatterbox`: `default`. `veena`: `kavya`, `agastya`, `maitri`, `vinaya`, ... (an unknown voice answers 400 listing the valid ones) |
| `language` | `chatterbox-mtl` only: ISO 639-1, one of ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh (default `en`) |
| `response_format` | `wav` (default; 16-bit mono WAV) or `pcm` (raw s16le mono) |
| `stream` | `true`: chunked transfer; audio bytes arrive as they are synthesized |
| `seed` | optional integer; sampling is otherwise random per request |
| `speed` | only `1.0` |

Output is 24 kHz mono 16-bit for every TTS model. For streaming playback use
`response_format: "pcm"` with `stream: true` and play the bytes as they arrive
(`clients/speak.py` writes them to a WAV while streaming and reports time to first audio).

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"chatterbox-mtl","input":"Hello from the voice stack.","voice":"default"}' -o hello.wav
curl -sN http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"chatterbox-mtl","input":"Bonjour tout le monde.","voice":"default","language":"fr","response_format":"pcm","stream":true}' -o bonjour.pcm
```

Length per request: in `voice-core`, keep `chatterbox-mtl` input to **at most ~400 characters**
(2-3 sentences). Its text and speech share a 1024-token context there; longer input comes back
with the audio cut short (no error), and input past ~1500 characters is refused with 400
`context_length_exceeded` (KNOWN_ISSUES.md). Send long text sentence by sentence, as
`clients/voice_agent.py` does. `veena` speaks long input (up to 4096 characters) as consecutive
segments of whole sentences.

Admission: TTS streams are admitted while every playing stream can stay ahead of real time; a
request that would wait longer than 6 s gets 429 with `Retry-After`.

## LLM (Gemma-4 E4B)

### POST /v1/chat/completions

OpenAI chat completions. Served id `gemma-4-e4b`; context 8192 tokens (prompt + `max_tokens`;
longer is 400 `context_length_exceeded`). The checkpoint's chat template is applied (system,
user, assistant turns). Supported: `stream` (SSE, with `stream_options.include_usage`),
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`,
`logprobs` + `top_logprobs` (0-20), `chat_template_kwargs`. Refused with 400 rather than ignored:
`n != 1`, `tools` / `tool_choice` / `functions`, `response_format`. Defaults for unset sampling
fields come from the checkpoint's `generation_config.json`; pass `temperature: 0` for greedy.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
```

### POST /v1/completions

Raw text completion: the prompt is used verbatim, with no template and **no BOS added**. Gemma
prompts must start with `<bos>`; without it the output degrades. `prompt` may be a string or a
list of token ids. `logprobs` 0-20, `echo` refused.

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b","prompt":"<bos>The capital of France is","max_tokens":8,"temperature":0}'
```

## Sessions (voice agents)

Send the same `X-Session-Id: <id>` header on a call's ASR, chat and TTS requests. The server
keeps that session's state (ASR audio windows and prompt, LLM KV cache of the conversation so
far, TTS voice conditioning) for `SESSION_TTL_MS` (60 s) after each request, so the next turn
re-uses it instead of recomputing; `X-Request-Id` tags a request in logs and `Server-Timing`.
`clients/voice_agent.py` shows the full loop with per-stage timing.

## Limits summary

| | limit |
|---|---|
| JSON body | 64 MiB |
| transcription upload | 4 MiB, 0.5-30 s, 8-48 kHz |
| `/v1/audio/vad` upload | 32 MiB, up to 10 min |
| WebSocket / Realtime | 256 sessions; 32,000-byte PCM frames (native), 1 MiB events (Realtime) |
| concurrent transcription uploads | 256 (then 429) |
| TTS input | 4096 characters; chatterbox-mtl in voice-core: ~400 characters per request (see above) |
| LLM context | 8192 tokens (prompt + max_tokens) |
| HTTP connections | `HTTP_MAX_CONNECTIONS` (4096); idle keep-alive and header timeout 30 s |
| slow consumers | a client that stops reading a stream for 5 s is cut |
