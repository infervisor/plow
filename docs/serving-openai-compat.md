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

## 4. Known gaps, not fixed

- **Sampling is ignored on the AMD backend.** `temperature`/`top_p` are applied on CUDA only;
  the AMD engine samples the device argmax and logs one warning. Left as-is deliberately:
  refusing `temperature > 0` would break every existing client and benchmark, and the right fix
  is to wire the host resample path into the AMD arm.
- **Tool calling is refused, not implemented.** The templates can render tool blocks; nothing
  parses a tool call back out of the generation.
- **The handler projects each message to `{role, content}` before rendering**, so a template
  never sees `tool_calls`, `name` or `reasoning_content`, and `tools` is passed as none. Kimi's
  template keys the speaker off `message.get('name')`, and every family renders an assistant
  turn that carried a tool call as an empty one. Latent while `tools` is refused at the API
  boundary; it is the thing to fix first when tool calling lands.
- **Only two of the four places HF puts a template are read.** `ChatTemplate::load` reads
  `chat_template.jinja` and the `chat_template` string in `tokenizer_config.json`. It does not
  read `chat_template.json`, the `chat_templates/*.jinja` directory, or the list form of
  `chat_template` (`[{"name": "default", …}]`) — even though `nn-graph`'s `METADATA_FILES`
  stages the first two into the bundle. No checkpoint on this host uses them, so the gap is
  latent, but a checkpoint that does would fall through to the built-in builders.
- **`tojson` takes no keyword arguments.** GLM's template calls
  `tojson(ensure_ascii=False)` on the tool path, which minijinja rejects as too many arguments.
  Unreachable while tool calls are stripped before rendering.
- **No `/v1/embeddings`, no `/v1/models/{id}`.**
- **No auth, CORS, body-size limit or request timeout** on the router.
