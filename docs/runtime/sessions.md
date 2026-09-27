# Request and session identity on `plowrt serve`

Every OpenAI route (`/v1/chat/completions`, `/v1/completions`, `/v1/audio/speech`,
`/v1/audio/transcriptions` and its WebSocket) reads two optional headers:

| header | meaning |
|---|---|
| `X-Request-Id` | Names one request. Echoed on the response (and in every transcription SSE event, and the WebSocket `ready` event); generated (uuid v4 shape) when absent. Within a session, a request id already in flight there is refused with 409 (`duplicate_request_id`). No other dedupe. |
| `X-Session-Id` | Names a session of one model. After each of its requests finishes, the request's KV stays in its slot for the session's next request. Without it a request behaves exactly as before. |

Ids are 1..=128 visible ASCII characters (400 otherwise); header names are case-insensitive.
Both are echoed unchanged on every response, so a router or load balancer above can pin a session
to one server (session state is per process: nothing is shared between servers). A session that
lands on a server without its state (failover, restart, eviction, TTL) is served normally: its KV
is recomputed, never an error.

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
(`<|audio_end|><|im_end|>\n<|im_start|>assistant\n`) and the transcript are recomputed per partial.

The final is exact: it runs the full frontend and encoder over the whole recording, like a
one-shot request, and resumes only rows whose content matches. Cached window rows are not
bit-identical to a whole-utterance encode (the log-mel dynamic-range clamp is global, and the
encoder's split-K order follows the launch's single-utterance capacity), so the final reuses the
prompt prefix, not the windows.

The WebSocket route (`/v1/audio/transcriptions/stream`) runs its once-per-second partials the same
way on `plowrt serve`: each connection is a session (its `X-Session-Id`, or `ws-<request id>`).
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

A partial's cost is dominated by decoding its transcript (~2.2 ms/token, every partial from the
start: the partial's text rows are discarded), so the O(new audio) encoder and prefill take 8%
off the last partial's latency and 3% off the stream's total. Verifying the previous partial's
transcript as a draft (one prefill instead of token-by-token decode) is the remaining lever. Speech TTFA is set by the codec's first chunk; the conditioning prefix a
session saves is under a millisecond of prefill. A resumed prefill is not bit-identical to a
one-launch prefill (it is a chunked prefill split at the resume row): greedy Veena turn-2 text
matched the plain run 3/8 times, diverging within 0-3 audio tokens in 3 trials.
