# plowrt's OpenAI surface: what it serves, and what it refuses

Audited 2026-09-07 against the OpenAI API and against `zai-org/GLM-5.3`'s own
`chat_template.jinja`. Everything below is either fixed and covered by a test, or listed as a
known gap. `scripts/glm53_api_check.py` is the live battery: one case per defect found, run it
against any serving plowrt.

## 1. The chat prompt now comes from the checkpoint

plowrt had **no template engine**. Every chat prompt was built by a hand-written Rust function
per model family, chosen by probing the tokenizer for a family marker. Two failure modes, both
of which were live:

- A family matching **no** probe was served **Gemma's** markers, which its own tokenizer spells
  out as ordinary text. Fluent answer, wrong prompt, no error.
- A family that **did** match could still be wrong in detail. Checked against the rendered
  template for GLM-5.3, the GLM builder was wrong four ways:

| | the checkpoint's template | what plowrt emitted |
|---|---|---|
| generation prompt | `<\|assistant\|><think>` — thinking OPEN, 5.3's template has no disable branch | `<\|assistant\|><think></think>` — GLM-5.2's trick |
| system prefix | always emits `<\|system\|>Reasoning Effort: Max` | dropped |
| assistant history turn | `<\|assistant\|><think></think>{content}` | `<\|assistant\|>{content}` |
| `role: "tool"` | `<\|observation\|><tool_response>…</tool_response>` | rendered as a **user** turn |

`minijinja` now renders the template that ships with the weights.
`serve::template::ChatTemplate` reads `chat_template.jinja`, falling back to the
`chat_template` key inside `tokenizer_config.json`, from the asset dir or its `checkpoint/`.
A test loads the REAL checkpoint file and asserts the render is byte-identical to what
`transformers` produces. The hand-written builders remain the fallback for checkpoints that
ship no template — Kimi-K3 ships none — and an unrecognised family now logs an error instead of
silently getting Gemma's format.

**HF templates are written for Jinja2 on Python and call real `str` methods.** GLM's calls
`content.strip()` on an assistant turn, which minijinja has no notion of, so a multi-turn
conversation failed to render at all until `strip`/`lstrip`/`rstrip`/`startswith`/`endswith`/
`lower`/`upper`/`split` were added through an unknown-method callback. That gap was invisible
to the type checker and to a single-turn smoke test; only the live battery caught it.

**They call real `dict` methods too, and that one was fatal.** `message.get('tool_calls')` is
how a template reads an optional key, and Gemma-4's and Kimi-K2.5's both do it on their first
pass over `messages` — Gemma-4 at `chat_template.jinja:239`, Kimi-K2.5 at `:19`. minijinja has
no dict methods, so `render` failed for **every** conversation with `map has no method named
get`; and because the template itself COMPILED, `ChatTemplate::load` returned `Some` and the
built-in builders were never reached. The server answered **400 to every chat request** for
those families. `get`/`items`/`keys`/`values` are now handled by the same unknown-method
callback, with `get` returning Python's `None` for a missing key so the `a.get(x) or a.get(y)`
idiom stays falsy.

**And they call `strftime_now`, which is worse when it is *guarded*.** Templates stamp the
current date into the system prompt with `strftime_now(fmt)` — `datetime.now().strftime(fmt)`
on the `transformers` side, so local time, not UTC. Three families need it and each broke
differently without it:

| family | how the template calls it | what plowrt did |
|---|---|---|
| gpt-oss | unguarded, `chat_template.jinja:202` | render died, `undefined is not callable` → **400 on every request** |
| Llama-3.2 | `{% if strftime_now is defined %}` … `{% else %}"26 Jul 2024"` | rendered, telling the model **`Today Date: 26 Jul 2024`** |
| Muse-Glimmer | `{%- elif strftime_now is defined -%}` | rendered, **dropping the `Current date:` line** |

The two guarded ones are the dangerous pair: no error anywhere, and the model is simply told
the wrong day. It is now an env function backed by `chrono::Local`, with an unknown specifier
returned as a render error rather than the panic `to_string()` on a failed chrono `Display`
would raise inside a request handler. Llama-3.1 and Llama-3.3 were unaffected — their templates
hardcode the date string instead of calling out for it.

Rendering is verified against a Jinja env configured the way `transformers` configures it
(`ImmutableSandboxedEnvironment`, `trim_blocks`, `lstrip_blocks`, `loopcontrols`,
`raise_exception`, `tojson`), over six conversation shapes — user-only, system+user,
multi-turn, `developer` role, a tool result, and a null-content assistant turn.
`scripts/chat_template_check.py` is that diff, runnable against any models root
(`cargo build -p plowrt --example tmpl_probe` first); the two real-checkpoint tests in
`serve::template` pin GLM-5.3 and Gemma-4 where the weights are present.

| checkpoint | template | render |
|---|---|---|
| `zai-org/GLM-5.3` (`glm_moe_dsa`) | `chat_template.jinja` | 6/6 byte-identical |
| `zai-org/GLM-5.2` (`glm_moe_dsa`) | `chat_template.jinja` | 6/6 byte-identical |
| `google/gemma-4-12B-it` (`gemma4_unified`) | `chat_template.jinja` | 6/6 byte-identical |
| `google/gemma-4-26B-A4B-it` (`gemma4`) | `chat_template.jinja` | 6/6 byte-identical |
| Kimi-K2.5 (`kimi_k25`) | `chat_template.jinja` | 6/6 byte-identical |
| Kimi-K3 | ships none | built-in `k3_chat_prompt` |
| `openai/gpt-oss-20b` (`gpt_oss`) | `chat_template.jinja` | 5/5 byte-identical¹ |
| Llama-3.1, Llama-3.2, Llama-3.3 (`llama`) | `.jinja` / `tokenizer_config.json` | 6/6 byte-identical each |
| `meta-models/Muse-Glimmer-30B` (`muse_glimmer`) | `chat_template.jinja` | 6/6 byte-identical |

¹ gpt-oss's sixth shape — a `tool` message with no preceding assistant tool call — is refused by
the template's own `raise_exception` on both sides, which the server answers as a 400 carrying
the template's message. That is the designed path, not a divergence.

Template support is not serving support. `gpt_oss` and `llama` compile (`devgen`'s `gptoss.rs`
emitter and the dense-GQA path respectively; `llama` also has an `nn-graph` builder), and both
load a stock `tokenizer.json` through the `tokenizers` crate with no per-family fix-up — only
Qwen2 needs one. Every marker their templates emit resolves to ONE id, which is the property
that makes a rendered prompt real rather than literal text: gpt-oss `<|start|>` 200006,
`<|message|>` 200008, `<|end|>` 200007, `<|channel|>` 200005, `<|return|>` 200002 (the ids
`harmony_chat_prompt` documents); Llama `<|begin_of_text|>` 128000, `<|start_header_id|>`
128006, `<|end_header_id|>` 128007, `<|eot_id|>` 128009. `cargo build -p plowrt --features
hf-tokenizer --example tok_probe` checks that for any checkpoint. `muse_glimmer` is **refused by the compiler** (`nn-graph`'s config parser
rejects it, and `devgen` has no arm), so its template rendering correctly is moot until an
emitter exists. Likewise `kimi_k25` — the two Kimi-K2.5 checkpoints on this host hit the
`other =>` arm and cannot be built, and Gemma-4-26B-A4B is refused as MoE.

**Reasoning is split from the answer.** With thinking left open the raw generation is
`<trace></think><answer>`, and `</think>` is `special: false` in GLM's added tokens, so
`skip_special_tokens` does NOT remove it. `content` now carries the answer alone and the trace
goes to `reasoning_content`, on both the buffered and the streamed path.

**The asset bundle must carry the tokenizer-side files.** It used to symlink `tokenizer.json`
alone, so the template was unreachable, the Qwen2 pre-tokenizer fix-up could never fire, and
the eos set came from the `config.json` fallback rather than `generation_config.json`.

## 2. The endpoint fixes

**P0, silent corruption**

- **A streaming error was delivered as assistant text.** `delta.content = "[error: …]"` with
  `finish_reason: "stop"` on a **200**. openai-python, LangChain and `vllm bench serve` all
  score that as a successful request and hand the error to the user as if the model said it —
  and the causes are ordinary (no free slot, arrival-rate shed, KV OOM, context overflow,
  device fault). It is now an error object in its own SSE frame with no `[DONE]`.
- **`created` was absent** from every response, chunk and model card. Now stamped once per
  request and repeated across a stream.
- **`finish_reason: "preempted"`** is not a legal OpenAI value; a typed client rejects the whole
  response. Mapped to `"length"`, with the real cause in `x_plow_finish_reason`. It is NOT
  collapsed to `"stop"` — that would be a truncated answer claiming to be complete.

**P1, client-visible**

- Every error now uses the `{"error": {message, type, code}}` envelope. It used to be a bare
  `{"error": "<string>"}`, which openai-python cannot read.
- **Context-length overflow is 400 `context_length_exceeded`**, not 429. Seven raise sites
  across CUDA, AMD, the serve layer and the CPU path used to answer 429 or 500 — a 429 makes
  every OpenAI client retry a request that can never succeed.
- `stop` strings implemented, matched on streamed TEXT (a stop sequence need not be a token and
  can straddle a token boundary), suppressed under `ignore_eos` so benchmarks are unaffected.
- `seed` implemented, mixed into the sampling draw.
- `/v1/completions` accepts all four OpenAI prompt forms; batches are refused explicitly.
- `messages[].content` may be `null`, so an assistant tool-call turn can be replayed.
- Malformed JSON returns the envelope, not axum's plain-text rejection.
- **Refused rather than silently dropped**: `n != 1`, `functions`, `function_call`,
  `response_format`, `echo`, `suffix`, `best_of`, and `tools` wherever §2b cannot honor them.
  (`logprobs` / `top_logprobs` were on this list; they are served on the CUDA engine since
  2026-09-27, see below.) A silently dropped `tools` is a confidently wrong answer that scores
  as success.

## 2b. Tool calling

`tools`, `tool_choice` (`"auto"` / `"none"`), `parallel_tool_calls`, assistant `tool_calls` and
`role: "tool"` results are served on `/v1/chat/completions`, streamed and not, for any model whose
own chat template renders tools in a call syntax the server parses (`crates/plowrt/src/serve/tools/`).
Nothing is keyed on the model name.

**Is it supported?** Decided per model at load: the template is rendered once with a probe tool
(`ToolSupport::probe`). A template that never prints it ignores `tools` (Mixtral, DeepSeek-V3.x
whose templates have no tools block) and the request is refused (400, `param: tools`) instead of
answered without them. A model served without a template (built-in builders: Kimi-K3, DeepSeek-V4
with its Python-only `encoding_dsv4.py`) refuses `tools` and assistant `tool_calls` history.

**Call syntax**, read from the template's own markers (`ToolFormat::detect`), with one streaming
parser each:

| format | families (template checked) | model output |
|---|---|---|
| `gemma4` | Gemma 4 E4B / 12B / 26B / 31B | `<\|tool_call>call:NAME{k:<\|"\|>v<\|"\|>,n:1}<tool_call\|>` |
| `hermes` | Qwen3, Qwen2.5, Hermes | `<tool_call>{"name": .., "arguments": {..}}</tool_call>` (after any `<think>` trace) |
| `qwen3_xml` | Qwen3.5, Qwen3-Coder | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`; values typed by the tool's JSON schema |
| `glm45` | GLM-4.5 / 4.6 / 5 / 5.3 | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` |
| `llama3_json` | Llama 3.1 / 3.2 / 3.3 | an answer that opens with `{"name": .., "parameters": {..}}` (or `<\|python_tag\|>`), `;`-separated |
| `mistral` | Mistral v0.3 (`[TOOL_CALLS] [..]`) and v11+ (`[TOOL_CALLS]NAME[ARGS]{..}`) | ids are 9 alphanumerics, as the template requires |
| `kimi_k2` | Kimi-K2 | `<\|tool_call_begin\|>functions.NAME:IDX<\|tool_call_argument_begin\|>{..}`; the id is kept |
| `harmony` | gpt-oss | `commentary to=functions.NAME` messages; `analysis` → `reasoning_content` |

A template that renders tools in any other syntax refuses `tools` (400) rather than returning
unparsed calls as text.

**Request mapping** (`tools::request`), matching vLLM's hand-off to `apply_chat_template`:
`tools` is passed through unchanged; `tool_calls[].function.arguments` strings become objects
(a template that concatenates strings, DeepSeek's, is re-rendered with the original strings);
a tool-call turn's `content: null` becomes `""` (gpt-oss's template fails on `None`, GLM-4.5's
prints the word `None`); `tool_call_id`, `name`, `reasoning_content` pass through. Validation:
function tools only, names `^[a-zA-Z0-9_-]{1,64}$`, unique; `tool` messages need
`tool_call_id`; history `tool_calls` need `id`, `function.name` and JSON-object `arguments`.
The renderer now matches `transformers`' Jinja environment: `trim_blocks`/`lstrip_blocks`,
insertion-ordered maps, `loop.previtem`/`nextitem`, `none is iterable` false, `tojson` as
Python's `json.dumps` (`", "` / `": "` spacing, `ensure_ascii`, `indent`, `separators`,
`sort_keys`, float `repr`), and `strip`/`split`/`replace` with Python's arguments.

**Refused** (400): `tool_choice: "required"` and a forced function (no constrained decoding, so
the call cannot be guaranteed); the deprecated `functions` / `function_call` (use `tools`);
non-function tool types. `tool_choice: "none"` renders the conversation without `tools` and
does not parse. `parallel_tool_calls: false` keeps only the first call. An empty `tools: []`
is the same as none.

**Response.** For these requests the generation is decoded with special tokens KEPT (the call
markers are special tokens in most vocabularies), split for reasoning, parsed, and only then
stripped of the remaining special tokens. `message.tool_calls` carries
`{id, type: "function", function: {name, arguments: <JSON string>}}`, `content` is the text
before the calls or `null`, and `finish_reason` is `"tool_calls"` when the turn ended on its own
(a turn cut by `max_tokens` stays `"length"`). Streaming sends, per completed call, one delta with
`index`, `id`, `type`, `name` and empty `arguments`, then one with the full `arguments`; markers
never reach `delta.content`. A call is emitted once it is complete, so arguments arrive per call,
not per token. A malformed or truncated call falls back to text, stripped as before.

**Parity.** `scripts/llm/toolcall_fixtures.py` renders 7 conversations per family with
`transformers` (`render_jinja_template`, the `apply_chat_template` code path) into
`crates/plowrt/tests/fixtures/toolcall/`; `serve::tools::parity_tests` renders the same
OpenAI-shaped requests through the handler's mapping and must match the text, and the token ids
where the family's `tokenizer.json` is on the host. 15 families: gemma4-e4b, gemma4-12b, qwen3,
qwen2.5, qwen3.5, qwen3-coder, llama3.1, llama3.2, mistral-v0.3, glm4.5, glm5.3, kimi-k2,
gpt-oss, and the two refused (deepseek-v3.1, mixtral).

**No packet re-emit is needed**: the format comes from the chat template the packet already
carries in `serve.json`.
- `/health` added alongside `/healthz`; `/tokenize` reports `count`; model cards carry
  `created`; request ids are seeded per process instead of starting at zero; CUDA reads the
  same stop-id sources as the AMD and CPU engines.

## 2c. Images and audio (`PLOW_EMIT_MULTIMODAL=1` packets)

A packet built with `PLOW_EMIT_MULTIMODAL=1` carries one encoder sidecar per tower in the
checkpoint (`mm_vision.pkt`, `mm_audio.pkt`). Each is a `forward.v1` pipeline `mm.encode` with a
rung ladder: `PLOW_EMIT_MM_VISION_LADDER` (images per launch, default `1,2`) and
`PLOW_EMIT_MM_AUDIO_LADDER` (log-mel frames, default `400,1000,2000,3000`). `model.pkt` also gets
a `plow.multimodal.v1` section (placeholder ids, processor parameters, slab size). plowrt reads
only that metadata. It has no per-model code, so adding a model means emitting its towers.

- **Request parts.**
  - `image_url` and `input_image` accept `data:` URLs only. `http(s)` URLs get 400; the server has
    no fetcher.
  - `input_audio` accepts `format: "wav"` (any rate, resampled; channels mixed to mono). Other
    formats get 400.
  - Text-only requests render the template exactly as before.
- **Unsupported.** A model without a matching tower answers 400
  `unsupported content type for this model: <kind>`.
- **Discovery.** `/v1/models` cards carry `x_plow_modalities` (`["text","image","audio"]`).
- **Preprocessing** runs on the CPU from the contract's parameters. Images get an
  aspect-preserving resize (Pillow bicubic, fixed point), then patches. Audio becomes a
  semicausal log-mel spectrogram. The template's placeholder expands to
  `begin + n × soft-token + end`, with n computed the way the HF processor computes it.
- **Injection.** Each soft token is a prompt id with bit 31 set, holding a 31-bit content hash of
  (kind, media bytes, row). The LM's `Embed` maps those ids to the pad row. `MmRowsBf16` then
  replaces each of those rows with the encoder's projected row, which a per-engine slab holds
  (`in.mm_slab` plus the hash table `in.mm_table`).
  - Rows are reserved at submit, staged before the launch and released when the job ends. A full
    slab answers 503.
  - Each engine instance (a model, or one DP rank) owns its slab and its encoders. Encoding runs
    after rank selection, on that rank's device. Both are dropped when the engine unloads.
  - Before launch, every bit-31 id in the prompt must be in the serving engine's table. Otherwise
    the request fails; it is never served the pad row.
  - Because the ids hash the media, the prefix cache and session keys see different images as
    different prompts. Token-batch, mixed-step and the VMM prefix cache all keep working.
- **Limits.** These answer 400: `PLOW_MM_MAX_IMAGES` (8), `PLOW_MM_MAX_AUDIO` (4),
  `PLOW_MM_MAX_IMAGE_PIXELS` (40M) and `PLOW_MM_MAX_AUDIO_SECONDS` (30).
- **Streaming, logprobs and tools** are unchanged: media only changes prompt ids.
- **Per model (Gemma 4).**

  | model | image | audio |
  |---|---|---|
  | E4B | yes: gemma4_vision tower | yes: USM conformer |
  | 12B | not emitted | yes: encoder-free 640-sample frames → `embed_audio` |
  | 26B-A4B, 31B | not emitted | none in the checkpoint |

  - Vision is skipped (with a logged reason) on every checkpoint whose text config sets
    `use_bidirectional_attention: "vision"`. Those LMs attend bidirectionally within each image on
    their sliding layers, and the LM attention kernels are causal-only.
  - 26B/31B vision also needs head_dim 72 in the tower's attention, which supports only 64 and 128.

## 3. Refusing to serve from the CPU by accident

plowrt warned and continued when a bundle's target GPU had no matching driver, then served it
from the CPU reference interpreter: stand-in logits, a bare `role:\ncontent` prompt flatten,
and newline-byte stop matching. Fluent, fast, wrong, and indistinguishable from a working
server unless someone reads the log — which is how a GLM-5.3 serve here came up on the CPU
backend and answered "The capital of France is Paris." at fictional speed. A bundle carrying a
compiled device blob with no matching driver is now a hard refusal at startup. A bundle with no
blob is a genuine CPU-reference asset and still only warns.

## 4. Sampling, per-model config and reasoning framing

Audited 2026-09-19. Six defects of one shape: a request field parsed away, or probed for in the
wrong place, and answered with a 200 under different behaviour than it asked for.

**`seed` reached only the path with no model.** `GenParams.seed` was plumbed to the slot, and
exactly one call site passed it on — the no-bucket reference fallback. The CUDA device sampler,
the host resample after a logits download, the batched CPU step and `step_token` all called the
seedless `seeded_unit`, so a fixed `seed` changed nothing on any real serve. One `slot_rng01`
helper owns the draw now and every sampling site goes through it.

**The sampler had six knobs; the API parsed two.** `top_k`, `min_p`, `repetition_penalty` and
`logit_bias` were implemented in `text::sample` and dropped by serde as unknown fields.
`presence_penalty` and `frequency_penalty` are OpenAI-standard, were also dropped, and were not
implemented either — they are additive and count-weighted, not the multiplicative
`repetition_penalty`, so they are their own pass over the row's history. All six are parsed and
applied, along with vLLM's `min_tokens` and `stop_token_ids`.
`SamplingParams::needs_host_logits` is now the single predicate deciding device vs host
sampling, so the eligibility checks cannot drift from the knob set again.

**Nothing was range-checked.** `top_p: 0` truncates the candidate set to nothing and a negative
`temperature` falls through the `<= EPSILON` greedy branch into an inverted softmax. Every field
is validated and an out-of-range value is a 400 naming it, as OpenAI answers.

**`chat_template_kwargs` did not exist here.** `{"enable_thinking": false}` is how every
Qwen3/GLM client turns reasoning off, and `scripts/bench_vllm_rocm.sh` sends exactly that to
vLLM — so both sides of that A/B were rendering different prompts, and plowrt's numbers carried
a thinking trace vLLM's did not. The render context is merged now rather than fixed, so any
variable a template reads can be supplied; `reasoning_effort`, `continue_final_message` and a
per-request `chat_template` ride the same `RenderOpts`.

**The checkpoint's sampling defaults were never read.** `generation_config.json` was parsed for
the eos set only, so every model was served at the stock temperature 1.0 / top_p 1.0 whatever
its authors chose (Qwen3 ships 0.6/0.95/20). `serve::config::ServingConfig` resolves it per
model at load, and resolution is now **request > model default > server default**.

**`opens_reasoning` searched the whole rendered prompt for `<think>`** — and the rendered prompt
contains user text. Asking "what does the `<think>` tag do?" routed the entire answer into
`reasoning_content` and returned an **empty `content`**, on any model, reasoning or not.

The framing is now a `ReasoningMode` decided from the rendered prompt's **suffix**, never a
search of it: `add_generation_prompt` puts the generation prompt last, so only a marker at the
very end is the model's own, and user text can no longer reach it. It is decided PER REQUEST
rather than per model, deliberately — it is the rendered prompt that decides, so one rule covers
the checkpoint's own template, the built-in builders for checkpoints that ship none, and a
request that turned thinking off with `chat_template_kwargs` (which renders the pair CLOSED and
so correctly reads as no trace).

The mode also fixes two disagreements between the two response paths: a trace that opened and
never closed is now reported as all-trace on both (the buffered path used to call it the
answer), and the streamed path FLUSHES its held tail on the terminal chunk. While inside a trace
the router withholds the last few bytes in case they begin the close marker; when generation
ended first those bytes were simply dropped, so streamed `reasoning_content` came back up to
`len("</think>")-1` bytes shorter than what the model produced.

## 5. The catalogue, usage and metrics

- **`GET /v1/models` stamped `created` with `now_secs()` per card**, so it changed on every
  scrape and no client could treat it as an identity. It is the process start, stamped once.
- Cards carry **`max_model_len`** (captured at engine install, so a card never takes the engine
  mutex behind a live tick), `root`, `parent` and `permission`, and **`GET /v1/models/:id`**
  exists. That route is GET-only with a 404 fallback: the admin routes live under
  `/v1/models/`, and a bare `get(...)` answered a POST to `/v1/models/load` with 405 on the
  PUBLIC router, announcing a control plane that listener does not serve.
- **Model aliases** (`--served-model-name ALIAS[=SLUG]`, and `aliases` on the admin `load`).
  Resolved once at the top of each handler so every slug-keyed map — metrics, residency, mux —
  keeps one key per model; the response echoes the name the client sent, as vLLM does. Aliases
  appear in the catalogue with `parent`/`root` naming their target, and are dropped when their
  target unloads.
- **`usage.completion_tokens_details.reasoning_tokens`** is reported for models that frame a
  trace, counted from chunks on both paths — a character split cannot be converted back into a
  token count.
- **`vllm:num_preemptions_total`** and the **`vllm:` prefix-cache pair** join the existing
  `vllm:` block. The prefix pair counts attach REQUESTS and its HELP says so: vLLM counts tokens
  queried and tokens hit, this server never counts the tokens it MISSED, and a token ratio here
  would be fabricated.
- **The request body limit is explicit** (64 MiB). axum's default is 2 MiB, applied by the
  `Json` extractor before any handler runs, so a long-context conversation was refused with a
  bare 413 and no error envelope.

## 6. Template loading

All **four** places HF puts a template are read now: `chat_template.jinja`, `chat_template.json`,
the `chat_template` key in `tokenizer_config.json` **as a string or as the list form**
(`[{"name": "default", …}]`), and the **`chat_templates/` directory**. A checkpoint using one of
the last two used to fall silently through to the hand-written builders.

`tojson` **takes keyword arguments**. GLM's tool path calls `tojson(ensure_ascii=False)`, which
minijinja rejected as too many arguments, failing the whole render. It is now Python's
`json.dumps` byte for byte, `ensure_ascii=True` included (§2b).

## 6b. Stop strings, reasoning framing and empty conversations

Found by serving real checkpoints (Qwen3-0.6B, SmolLM2-360M, Gemma-4-E2B on an M4 Pro) rather
than by a test.

**`stop` corrupted its own output, two ways.** The hold, the release and the cut were three
steps in a row over different strings:

- a token that withheld a tail met the release block in the SAME call and got those bytes
  prepended back in front of what remained, so `stop: ["three"]` over `"Count: "` emitted
  **`"tCoun"`**;
- and the cut is an index into THIS token's delta while it was being applied to the combined
  run, so a match that STARTS inside the withheld bytes re-emitted them — `stop: ["France is"]`
  over a generation of `" France is …"` returned **`" France"`** instead of `" "`.

`apply_stop_strings` now owns all three steps over one ordered run of generated-but-unemitted
bytes, and is a free function precisely so the SEQUENCING can be driven token by token in a
test. Both helpers it calls were already unit-tested and were already correct; nothing exercised
the order they were called in.

**A reasoning trace whose opening marker is not in the first token.** The splitter settled on
leading whitespace, so a model emitting `"\n"` then `"<think>"` never opened a trace at all and
the whole thing — marker included — reached `content`. Qwen3 happens to emit `<think>` as one
whole first token; nothing guarantees that tokenization. The splitter now waits while the
generation is still all whitespace.

**An empty conversation was answered.** `messages: []` rendered a bare generation prompt and the
model invented a question and answered it, with a 200. OpenAI's schema requires at least one
message; `/v1/completions` already refused its empty-prompt equivalent. Now a 400.

**Template rendering was verified, not assumed.** `scripts/chat_template_check.py` against
Qwen3-0.6B, SmolLM2-360M and Gemma-4-E2B: 18/18 conversation shapes byte-identical to
`transformers`' own Jinja2, including the `developer` role, a tool result and a null-content
assistant turn.

## 7. Known gaps, not fixed

- **Sampling is applied on CUDA ONLY.** The gfx950 engine and the CPU/Metal engine both sample
  on device and never hand the host a logit row, so every token is the argmax whatever
  `temperature`, `top_p`, `top_k`, the penalties or `seed` asked for. The host resample
  (`gpu_finish_token`) is reached only from `cfg(cuda)` call sites, and neither
  `serve::cpu_serve` nor the Apple engine contains a sampler.

  The AMD half was already documented. The CPU/Metal half was not, and was found by serving a
  real Qwen3-0.6B on an M4 Pro: five different `seed`s, and `temperature` 0, 0.7 and 2.0, all
  returned byte-identical text. Still deliberate — refusing `temperature > 0` would break every
  existing client and benchmark, and the real fix is to wire the host resample path into the
  shared single-sequence tick. It is no longer only a once-per-process log line:
  `ServeEngine::honours_sampling` is captured at install and reported as
  `x_plow_sampling: "device_argmax"` on the model card, so a client can see it before it sends a
  request whose sampling will be discarded.

  The per-model sampling DEFAULTS above are still read and still plumbed; they simply have no
  effect on these two backends yet, and the card says so.
- **Tool calling** is served per §2b. Not served: `tool_choice` `required` / forced function,
  incremental per-token `arguments` streaming, DeepSeek's `<｜tool▁calls▁begin｜>` and DSML
  syntaxes (no DeepSeek checkpoint ships a template that renders `tools`), and Hermes-3's named
  `tool_use` template (the default template is picked).
- **Only think-tag reasoning is parsed** (outside tool requests; a gpt-oss tool request parses
  its harmony channels, §2b). `ReasoningMode` covers `<think>`/`</think>`, which is
  how GLM, Qwen3 and DeepSeek-R1 frame a trace. gpt-oss's harmony channels are NOT parsed and
  its trace still reaches `content`. A harmony arm was written and then removed rather than
  shipped: the built-in `harmony_chat_prompt` pins the FINAL channel (reasoning off), so a mode
  that assumes a trace is open from token zero would route the whole answer into
  `reasoning_content` and return an empty `content` — the exact failure this section exists to
  remove — and there is no gpt-oss checkpoint on this host to settle what the real template
  renders. The mode enum is the seam: a verified harmony arm is a variant and a close marker.
- **`logprobs` is served on the CUDA engine only** (chat `logprobs`/`top_logprobs`, completions
  `logprobs` 0..=20, plus vLLM's `logprobs_mode` `raw_logprobs`|`raw_logits` per request and
  `return_tokens_as_token_ids`); other backends refuse it with 400. Values are the raw model
  distribution after the checkpoint's final-logit softcap. A request that asks for them samples on
  the host (one logits-row download per token) and leaves the device multi-step quantum.
  `echo`/prompt logprobs are still refused. Contract and parity: `docs/runtime/gemma4-e4b-h100.md`.
- **No `/v1/embeddings`.**
- **No CORS and no TLS** on the router. API keys are optional (`--api-key`, repeatable, or
  `PLOW_API_KEYS=k1,k2`): when set, every route except `/health` and `/healthz` requires
  `Authorization: Bearer <key>` or `x-api-key: <key>` (401 otherwise), on TCP and on the admin
  UDS. Without keys the body-size limit and the UDS's mode 0600 are the only access controls,
  and a non-loopback bind logs a warning. Put a TLS-terminating proxy in front of a public bind.
  A blanket request timeout is deliberately absent for text, because one would cut legitimate
  long generations; transcriptions have one (`--asr-request-timeout-ms`, 120 s, 504).
- **Reasoning traces spend `max_tokens`.** There is no separate budget for the trace, so a
  reasoning model with a small cap can be truncated before its answer. `chat_template_kwargs`
  now gives clients the same escape hatch they use against vLLM, but a real reasoning budget is
  a scheduler feature, not an API one.
- **No per-model fairness.** Co-tenants on a device group take turns through
  `cosched::DeviceTurn` with no weight or priority.

## 7b. Running it in production

- **Run plowrt under a supervisor** (systemd `Restart=always`, a Kubernetes pod) that restarts it
  when it exits. The release build is `panic = "abort"`: any panic ends the process instead of
  one request, and nothing inside plowrt brings it back.
- **Probe `/health` (or `/healthz`).** It answers 503 once an engine is dead (a fatal device
  fault poisons the context and every later request fails), so the orchestrator can restart the
  instance or route away from it.
- SIGTERM/SIGINT turn `/health` to 503, stop admission and drain live generations (bounded by
  `PLOW_DRAIN_TIMEOUT_MS`, 30 s when unset) before exiting 0.
- Connection limits: `PLOW_HTTP_HEADER_TIMEOUT_MS` (request head and idle keep-alive, 30 s) and
  `PLOW_HTTP_MAX_CONNECTIONS` (4096 per listener). A consumer that stops reading a stream is
  parked for up to 5 s, then cut with "response consumer is too slow".
- **Transcription** (`POST /v1/audio/transcriptions`, WebSocket
  `/v1/audio/transcriptions/stream`) is served for every audio-LM bundle (Qwen3-ASR, CUDA)
  and every `--asr-packet` model (Nemotron RNNT on its own cohort engine); `/v1/models` lists
  those endpoints on its card. WebSocket `mode: continuous` streams unbounded audio as
  endpointed segments, one `final` each. Limits: 4 MiB body, 0.5 to 30 s of 8 to
  48 kHz WAV (resampled to 16 kHz), 256 concurrent uploads and 256 WebSocket sessions. A full
  queue answers 429 with `Retry-After: 1`; shutdown and a closed dispatcher answer 503; a missed
  deadline answers 504. On SIGTERM `/health` turns 503 first, new transcriptions are refused,
  and WebSocket sessions still receiving audio end with close code 1001. Protocol, fields and
  status table: `docs/runtime/asr.md`. `GET /v1/realtime?intent=transcription` serves the same
  models over OpenAI's Realtime transcription-session protocol (base64 pcm16 / G.711 audio,
  server VAD or manual commits, transcription delta/completed events).

## 8. Unrelated, found while doing the above

`cargo check -p plowrt --features cpu` does not compile on `main`, independent of any of this:
`serve/mux.rs` calls `RuntimeConfig::get().multistep()`, which is `#[cfg(any(cuda, hsa))]`, from
an AMD multistep block that is not itself hsa-gated. Not touched here.

### CPU/Metal model lifecycle and compiler identity

Native CPU/Metal slot engines support the existing private `/v1/models/load`,
`/v1/models/unload`, and `/v1/models/status` API, including explicit asset
registration, aliases, deregistration and preemption of active streams. Status
advertises `capabilities` so a controller can distinguish this support from a
backend without a lifecycle manager. Automatic eviction is not supported on
CPU/Metal. An unloaded model remains registered until `deregister: true`, and
inference returns `model_unloaded` until an explicit load. Controls finish after
client disconnect; public TCP inference does not expose these admin routes.
Native allocation reclamation counters are `null` when they cannot be measured.

`/v1/models` cards advertise `x_plow_endpoints`; text slot models currently list
`chat/completions` and `completions`. Use private status to distinguish registered,
resident and serving models. Resolve a card's `root` before issuing canonical
model controls; aliases share residency and the same control lock.

The compiler writes the network slug and optional serving identity to
`weights.json`. Both packet and device-blob compilation honor `--served-name`.
For a Hugging Face cache snapshot, the network slug comes from the model name,
not the snapshot revision; the default serving identity preserves the HF repo
ID (including organization and case). An ordinary local directory defaults to
its lowercase basename. Output directories do not determine API identity.
`plowrt` precedence is an explicit runtime slug, then manifest `served_name`,
then manifest `network`; alternate client names should use aliases.

Standalone ASR accepts `--served-model-name nemotron-asr-0.6b` and publishes that
exact identity through `/v1/models` and `/v1/models/{id}`, with audio endpoint
capabilities. This separates a public model name from the packet pipeline's
operation name (often `transcribe`). Without the flag, the pipeline name remains
the compatibility default. Standalone ASR does not advertise text inference or
model lifecycle controls.
