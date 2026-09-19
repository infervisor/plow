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
- **Refused rather than silently dropped**: `n != 1`, `tools`, `tool_choice`, `functions`,
  `function_call`, `response_format`, `logprobs`, `top_logprobs`, `echo`, `suffix`, `best_of`.
  A silently dropped `tools` is a confidently wrong answer that scores as success.
- `/health` added alongside `/healthz`; `/tokenize` reports `count`; model cards carry
  `created`; request ids are seeded per process instead of starting at zero; CUDA reads the
  same stop-id sources as the AMD and CPU engines.

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
minijinja rejected as too many arguments, failing the whole render. `ensure_ascii=True` is
refused explicitly rather than answered with bytes that do not match the request, because
`serde_json` never escapes non-ASCII.

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
- **Tool calling is refused, not implemented.** The templates can render tool blocks; nothing
  parses a tool call back out of the generation.
- **Only think-tag reasoning is parsed.** `ReasoningMode` covers `<think>`/`</think>`, which is
  how GLM, Qwen3 and DeepSeek-R1 frame a trace. gpt-oss's harmony channels are NOT parsed and
  its trace still reaches `content`. A harmony arm was written and then removed rather than
  shipped: the built-in `harmony_chat_prompt` pins the FINAL channel (reasoning off), so a mode
  that assumes a trace is open from token zero would route the whole answer into
  `reasoning_content` and return an empty `content` — the exact failure this section exists to
  remove — and there is no gpt-oss checkpoint on this host to settle what the real template
  renders. The mode enum is the seam: a verified harmony arm is a variant and a close marker.
- **The handler projects each message to `{role, content}` before rendering**, so a template
  never sees `tool_calls`, `name` or `reasoning_content`, and `tools` is passed as none. Kimi's
  template keys the speaker off `message.get('name')`, and every family renders an assistant
  turn that carried a tool call as an empty one. Latent while `tools` is refused at the API
  boundary; it is the thing to fix first when tool calling lands.
- **`logprobs` is refused, not served.** vLLM serves it and lm-eval-harness needs it; this is
  the blocker for evaluation parity.
- **No `/v1/embeddings`.**
- **No auth and no CORS** on the router. The body-size limit and the admin listener's 0600 UDS
  are the only access controls; a request timeout is deliberately absent, because a blanket one
  would cut legitimate long generations.
- **Reasoning traces spend `max_tokens`.** There is no separate budget for the trace, so a
  reasoning model with a small cap can be truncated before its answer. `chat_template_kwargs`
  now gives clients the same escape hatch they use against vLLM, but a real reasoning budget is
  a scheduler feature, not an API one.
- **No per-model fairness.** Co-tenants on a device group take turns through
  `cosched::DeviceTurn` with no weight or priority.

## 8. Unrelated, found while doing the above

`cargo check -p plowrt --features cpu` does not compile on `main`, independent of any of this:
`serve/mux.rs` calls `RuntimeConfig::get().multistep()`, which is `#[cfg(any(cuda, hsa))]`, from
an AMD multistep block that is not itself hsa-gated. Not touched here.
