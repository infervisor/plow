# Request and session identity on `plowrt serve`

Every OpenAI route (`/v1/chat/completions`, `/v1/completions`, `/v1/audio/speech`,
`/v1/audio/transcriptions` and its WebSocket) reads these optional headers:

| header | meaning |
|---|---|
| `X-Request-Id` | Names one request. Echoed on the response (and in every transcription SSE event, and the WebSocket `ready` event); generated when absent. Within a session, a request id already in flight there is refused with 409 (`duplicate_request_id`). No other dedupe. |
| `X-Session-Id` | Names a session of one model. After each of its requests finishes, the request's KV stays in its slot for the session's next request. Without it a request behaves exactly as before. It also links the call's requests across models into turns (below). |
| `X-Turn-Id` | The turn (one user utterance → transcript → reply → speech) a request serves. Inferred when absent. |
| `traceparent` | W3C trace context. The trace id is propagated; each request gets its own span id. Invalid → ignored (`tracestate` is not read). |
| `X-Turn-Budget-Ms` | End of user speech → first agent audio target for the turn (default `--turn-budget-ms`, 1500). |
| `X-Playback` | TTS only: client playback state, `started=<unix ms>` or `buffered_ms=<n>`. Recorded on the turn. |

Request and turn ids are 1..=128 visible ASCII characters, session ids 1..=256 (400 otherwise);
header names are case-insensitive. Ids are echoed unchanged on every response, so a router or load
balancer above can pin a session to one server (session state is per process: nothing is shared
between servers). A session that lands on a server without its state (failover, restart, eviction,
TTL) is served normally: its KV is recomputed, never an error. Ids the server mints (request ids,
a WebSocket's session, inferred turn ids) are [svid](https://crates.io/crates/svid)s: 11-char
base58, time-sortable; client ids are never parsed.

### OpenRouter compatibility

Body fields route like OpenRouter's (first match wins). Response bodies are unchanged.

| field | sources, in precedence order |
|---|---|
| session | body `session_id` > `X-Session-Id` (`prompt_cache_key` is accepted and ignored) |
| request | `X-Request-Id` |
| trace | `traceparent` > body `trace.trace_id` + `trace.parent_span_id` (other `trace` keys ignored) |
| turn | `X-Turn-Id` > body `metadata.turn_id` |
| budget | `X-Turn-Budget-Ms` > body `metadata.turn_budget_ms` |

Chat, completions and speech take them in the JSON body; transcription uploads take `session_id`,
`turn_id` and `turn_budget_ms` form fields. An invalid body `session_id` is a
400; the other body fields are hints, ignored when invalid.

### Turns

A turn table (process-wide, all models; `serve/turns.rs`) links a call's stages. Without
`X-Turn-Id` the turn is inferred per session:

* an ASR `final=true` upload (or a WebSocket `finish`) opens a new turn; its arrival is the end of
  speech;
* the session's next chat/completion joins the open turn, and the TTS request after it joins and
  closes it;
* a chat or TTS with no open turn opens one at its own arrival (text agents, TTS-only load).

ASR partials (`append`) are no turn stage. A turn without a client trace id gets one minted (its
high half is the turn's svid), shared by all its stages. Turns live for the session TTL (60 s when
retention is off), 32 per session. `GET /v1/turns/{session}` returns a call's turns: per stage the
arrival, admission, first output and done times (ms from the end of speech), device and wait-turn
ms, the server's playback clock and worst underrun.

### Response headers and Server-Timing

Every response carries `X-Request-Id`, `X-Session-Id` (when set), `X-Turn-Id` (assigned or
echoed), `traceparent` (the turn's trace id, this request's span id) and a W3C `Server-Timing`:

| metric | meaning |
|---|---|
| `queue` | arrival → slot admission (session requests: the admission report) |
| `wait-turn` | admission → first output not spent in the model's own ticks: co-tenant device turns and host gaps |
| `device` | the model's tick time over the same interval |
| `first` | arrival → first output (first token, transcript, first audio) |
| `total` | arrival → done, when known |
| `slack` | stage target − `first` (negative = missed): ASR final 500, LLM TTFT 800, TTS TTFA 800 ms (`--turn-asr-final-ms`, `--turn-llm-ttft-ms`, `--turn-tts-ttfa-ms`) |
| `turn` | end of speech → this request's first output |

A streamed chat/completion sends what is known at header time (`queue`) in the headers and the
full set as a final SSE comment line before `data: [DONE]`: `: server-timing queue;dur=…, …`.
SSE clients ignore comment lines. A streamed transcription does the same; its events carry
`turn_id`, and the WebSocket `final` event carries `turn_id`, `traceparent` and `server_timing`.
A streamed speech response sends its headers with the first audio, so they carry its timing.

`/metrics` adds `plowrt_turn_stage_seconds{model_name,stage=asr_final|llm_ttft|tts_ttfa}`,
`plowrt_turn_response_seconds` (end of speech → first audio), `plowrt_tts_underrun_seconds`
(worst underrun per stream on the server's clock), `plowrt_deadline_slack_seconds{stage}` and
`plowrt_deadline_missed_total{stage}`. A `tracing` span per request (`request_id`, `session`,
`turn`, `trace_id`, `model`, `stage`) logs admission, first output and done at debug level.

Session responses also carry

| header | meaning |
|---|---|
| `X-Session-Cache` | `hit` (retained rows resumed), `miss` (none retained for the session, or none shared), `evicted` (its rows were retained and dropped: TTL, a live request, the KV budget or the cap), `prefix-cache` (the engine's VMM prefix cache serves sessions, see below), `off` (retention disabled). |
| `X-Session-Cached-Tokens` | Prompt rows resumed instead of prefilled. |

and `/metrics` exports, per model: `plowrt_session_hits_total`, `plowrt_session_misses_total`,
`plowrt_session_reused_tokens_total`, `plowrt_session_evictions_total`, and the gauges
`plowrt_session_retained`, `plowrt_session_retained_tokens`, `plowrt_session_retained_bytes`.

## Retained KV

A session request's slot is not freed when it finishes: its rows stay, keyed by the session,
until one of

* the session's next request is seated: it resumes from the longest prefix whose rows are
  identical and prefills only the rest (at least its last prompt row: the first token needs its
  logits), then retains its own rows in turn;
* the idle TTL passes (`--session-ttl-ms` / `PLOW_SESSION_TTL_MS`, default 60000; 0 disables
  retention);
* a live request needs the slot, or the KV budget needs its rows: the least recently used session
  goes (a retained slot never queues a live request);
* a live request would otherwise be seated past the decode width the live requests already run
  (decode launches cover every slot up to the highest live one): the least recently used retained
  slot below goes, so retention never widens a launch (`--session-slack` / `PLOW_SESSION_SLACK`,
  default 0, lets it push that many slots further). For the same reason a session resumes a slot
  above that width only when every slot below it is live; otherwise it recomputes at the lowest;
* more than `--session-max` / `PLOW_SESSION_MAX` sessions are retained on one model (default 0:
  bounded by the slots).

Eviction is invisible to the client: the next request of an evicted session recomputes. Sessions
are per model (each mux has its own table) and a request only ever resumes its own session's rows.

Rows are compared by content, not by id: each prompt row's key is its token id folded with the
bits of every host overlay row that replaces its embedding (Qwen3-ASR audio rows, both CFG
members' Chatterbox T3 conditioning rows). Two requests with the same placeholder ids and
different audio share nothing past the first differing row. Only prefill-embedded prompt rows
are kept for overlay/position-base packets (their last prompt row and generated rows are embedded
by the decode program); a plain LM keeps prompt + generated rows, so a chat's next turn resumes
through the previous answer.

Two mechanisms, one per KV layout, never both:

* Static per-slot KV (every speech packet, and LLM packets without the VMM prefix cache): the
  slot itself is retained, as above (`GpuEngine::slot_resume_supported`, `resume_slot`). Zero
  copies; `begin_slot` starts the next sequence at the kept row.
* VMM prefix cache (`docs/runtime/prefix-cache.md`; Hopper hybrid-BF16 Gemma 4 by default): the
  cache already publishes each finished sequence's prompt and output blocks
  (`PLOW_PREFIX_CACHE_OUTPUT`, on), and a next turn attaches them, so slots are not retained. A
  session's retire instead pins its published path for the TTL (`PrefixCache::pin`): eviction
  takes every unpinned leaf first and a pinned one only when nothing else is left, so a pin orders
  eviction and never refuses an allocation. The reuse shows in
  `usage.prompt_tokens_details.cached_tokens`; the header says `prefix-cache`. The prefix cache
  stays off for overlay packets (their rows are not keyed by ids).

Other backends echo and dedupe the headers and retain nothing.

## Streaming transcription (`/v1/audio/transcriptions`)

Form fields beyond OpenAI's:

| field | meaning |
|---|---|
| `stream=true` | Server-sent events: `transcript.text.delta` (`delta`) as the decoder produces text, then `transcript.text.done` (`text`, `language`, `final`). Errors arrive as an `error` event. Every event carries `request_id` (and `session_id`). |
| `append=true` | Needs `X-Session-Id`. Appends the uploaded WAV to the session's recording and answers a revisable partial transcript of all of it (`"final": false`). |
| `final=true` | Needs `X-Session-Id`. Appends the upload (optional here) and answers the final transcript of the whole recording (`"final": true`); the recording then starts over. |
| `offset` | With `append`/`final`: the samples (16 kHz) the client has sent in this recording before this upload. |

Chunks may be any length (the 0.5 s minimum applies to the whole recording). A recording holds at
most 30 s (413 past it) and lives for the session TTL (60 s when retention is off); appends of one
session run one at a time. Every session answer carries `"offset"`: the samples the recording
holds after it.

The audio is the one piece of session state a server cannot recompute, so the contract is:
send `offset` with each upload. When it does not match what the server holds (the recording was
lost to a restart, failover or the TTL, or an upload was dropped), the upload is refused with 409
`session_audio_offset` and `"expected_offset": N`; the client resends its audio from sample N
(from 0 for a lost recording) and continues. Without `offset` uploads are appended as they
come.

A partial does O(new audio) work: the Qwen3-ASR encoder is chunk-local (conv) and window-local
(attention, 8 s windows), so each window is encoded once, when the audio has passed its end, and
cached with the recording; a partial encodes only the open window and resumes the decoder rows
through the completed windows from the previous partial. The prompt suffix after the audio rows
(`<|audio_end|><|im_end|>\n<|im_start|>assistant\n`) is recomputed per partial.

The previous partial's transcript, less its last 4 tokens, is forced as prompt rows (local
agreement), so a partial prefills those rows once (resumed from the session's retained rows) and
decodes only the tail. Tokens are not decoded one at a time from the start again.

Under load the model stops re-transcribing every append: after a partial that took `t`, the
session's appends answer the last partial's text until `t × (1/duty − 1)` has passed
(`--asr-partial-duty`, default 0.5). Idle partials take 30-50 ms against a 1 s append cadence, so
they all run.

The final is exact: it runs the full frontend and encoder over the whole recording, like a
one-shot request, and resumes only rows whose content matches. Cached window rows are not
bit-identical to a whole-utterance encode (the log-mel dynamic-range clamp is global, and the
encoder's split-K order follows the launch's single-utterance capacity), so the final reuses the
prompt prefix, not the windows.

The WebSocket route (`/v1/audio/transcriptions/stream`) runs its once-per-second partials the same
way on `plowrt serve`: each connection is a session (its `X-Session-Id`, or a minted one).
The `ready` event's `session_id` is that id. The cohort engine (`plowrt asr`) keeps its whole-buffer
partials and refuses `append`/`final`.

## Speech (`/v1/audio/speech`)

With `X-Session-Id`, a session's requests resume the shared prompt prefix: Chatterbox's voice
conditioning rows (both CFG members, the pair's two slots), Veena's prefix and speaker tokens.
`scripts/tts/tts_bench.py --session PREFIX` runs each client thread as one session.

## Chat and completions

`/v1/chat/completions` and `/v1/completions` take the same headers. A plain LM retains prompt and
generated rows, so a multi-turn chat's next prompt (the previous prompt, the answer, the new
message) resumes through the previous answer up to the first token the re-tokenized text changes.
`scripts/bench/session_turns.py` measures second-turn TTFT with and without a session.

## Measurements (H100, 2026-09-27)

One client unless noted; Qwen3-ASR `asr-goal/v5` assets, Veena staged, Chatterbox repro2.

| | without session | with session |
|---|---|---|
| Qwen3-ASR WER, 73 clips (one-shot vs 1 s appends + final) | 0.03913 | 0.03913, 73/73 finals identical to one-shot |
| 28.8 s clip in 1 s appends: partial latency first / last | 31 / 244 ms (whole-buffer re-send) | 31 / 225 ms |
| same, per stream: frontend + encoder, prefilled rows, decoded tokens | 420 ms, 6087, 1511 | 307 ms, 2403, 1511 |
| Veena LM `/v1/completions` second turn (~960-row prompt) TTFT | 21.3 ms | 9.2 ms (932 rows resumed) |
| Chatterbox stream c1 TTFA, CER | 219.0 ms, 0.000 | 219.1 ms, 0.000 (35 rows resumed) |
| Veena stream c1 TTFA, CER | 101.9 ms, 0.011 | 101.6 ms, 0.011 (2 rows resumed) |

A partial's cost used to be dominated by decoding its transcript (~2.2 ms/token, every partial
from the start), so the O(new audio) encoder and prefill took only 8% off the last partial's
latency. Forcing the previous partial's stable prefix removes that decode
(`session_stream_bench.py`, `PLOW_ASR_PARTIAL_DUTY=1`, H100, 2026-09-28):

| | decode every partial from the start | forced prefix |
|---|---|---|
| 73 clips, 1 s appends: partial p50 / p90 / max | 46.2 / 90.1 / 208 ms | 27.5 / 32.8 / 38.7 ms |
| same: engine time, final WER | 24.0 s, 0.03913 | 12.8 s, 0.03913 (73/73 finals identical) |
| 28.8 s clip: partial p50 / p90, engine time per stream | 126 / 202 ms, 3.37 s | 31 / 36 ms, 0.78 s |

Partials stay revisable. 373 of 443 are identical to the from-scratch partials (2.7% of words
differ), and 357 of 443 agree with the final up to their last two words (388 from scratch). Speech TTFA is set by the codec's first chunk; the conditioning prefix a
session saves is under a millisecond of prefill. A resumed prefill is not bit-identical to a
one-launch prefill (it is a chunked prefill split at the resume row): greedy Veena turn-2 text
matched the plain run 3/8 times, diverging within 0-3 audio tokens in 3 trials.

### Turn-aware scheduling, voice call_sim (H100, 2026-10-02)

`scripts/voice/call_sim.py`, Qwen3-ASR + Gemma-4 E4B + Chatterbox on one GPU, all calls open at
t=0, 3 turns per call, 2 reps. p50/p95 ms; underrun = turns with > 100 ms playback underrun.
"pre" is the branch before turn-aware scheduling; "turns" is this change. Errors 0 in all runs.

| calls | run | ASR final | LLM TTFT | TTS TTFA | underrun |
|---|---|---|---|---|---|
| 50 | pre | 166-195 / 374-417 | 71-92 / 231-279 | 371-390 / 681-761 | 0/150 |
| 50 | turns | 130-134 / 522-569 | 34-40 / 301-365 | 397-405 / 670-727 | 0-2/150 |
| 100 | pre | 219-243 / 540-556 | 160-164 / 877-993 | 1014-1034 / 2237-2390 | 150-193/300 |
| 100 | turns | 303-340 / 863-965 | 321-424 / 951-1007 | 690-729 / 1203-1232 | 62-94/300 |
| 200 | pre | 313-324 / 992-1012 | 1193-1333 / 66471-76768 | 2368-2592 / 4354-5269 | 481-502/600 |
| 200 | turns | 984-991 / 2382-2688 | 1072-1090 / 2357-2798 | 1740-1761 / 3662-3925 | 558-560/600 |

Turn deadlines bound the 200-call LLM tail (p95 ~70 s → 2.4-2.8 s) and cut 100-call TTFA and
underruns, at the cost of ASR finals at 100/200 calls and a 50-call ASR final p95 just over the
500 ms SLO. Ranking ASR finals above near-miss work fixed 50-call ASR p95 (182-250 ms) but at
overload starved the speech pipeline (200-call LLM TTFT p95 4-12 s), with or without deferring
ASR partials, so finals rank above first outputs but below near-miss work. 200 calls is over the
device's capacity (~92% of turns underrun in every variant).
