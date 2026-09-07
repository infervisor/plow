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
- **No `/v1/embeddings`, no `/v1/models/{id}`.**
- **No auth, CORS, body-size limit or request timeout** on the router.
