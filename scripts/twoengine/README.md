# Two-engine harness — one client, both engines

Measures plow and vLLM with **the same client code**, on the same box, in the same session.

## Why this exists

`scripts/bench_speed.sh` says it in its own header:

> NUMBERS FROM HERE MUST NOT BE TABLED NEXT TO A vLLM NUMBER. Different client, unvalidated
> against the reference implementation.

That is the correct warning, and it is why this directory exists. A campaign headline is a
**ratio between two engines**; if each engine is measured by its own harness, the ratio partly
measures the harnesses. Here one file drives both, with `bench_speed.sh`'s metric definitions
copied verbatim so the numbers stay comparable to project history:

| metric | definition |
|---|---|
| TTFT | time to the first SSE delta carrying content (not the role frame) |
| ITL | inter-token latencies, so a p99 exists |
| TPOT | `(last_tok_t − first_tok_t) / (out_tok − 1)` |
| out tok/s | aggregate completion tokens / wall |

`perf-data/plow-gfx942/README.md` calls re-baselining both engines in one session "the single
most valuable missing measurement". This closes it.

## Files

| file | what |
|---|---|
| `client.py` | accuracy + TTFT/TPOT ladder + GSM8K (non-streaming) |
| `speed.py` | streaming throughput ladder + long-context ladder |
| `needle_gate.py` | long-prompt needle-retrieval gate |
| `run_plow.sh` | serve plow → gates → `client.py` |
| `run_vllm.sh` | serve vLLM → gates → `client.py` |
| `run_speed.sh` | serve either engine → long gate → `speed.py` |

## The gates, and why each exists

Each one has produced a confident wrong answer on this box at least once.

1. **GPU lock + no sibling `plowrt`.** `pgrep -x` is comm-exact and misses a renamed binary
   (`plowrt_stock`); `pgrep -f "plowrt serve"` self-matches the launcher. Use `pgrep '^plowrt'`.
2. **HSA backend assert.** The nix flake does not carry `/opt/rocm-*/lib`, so
   `dlopen libhsa-runtime64.so.1` fails, plowrt falls back to the CPU reference interpreter, and
   it **serves perfectly** — correct answers, fictional timings. No output-based gate can catch
   this. `LD_LIBRARY_PATH` must be exported **inside** the nix shell (an inner `bash -c`);
   setting it outside does not survive `nix develop`.
3. **Coherence.** A fast wrong server is not a result.
4. **Needle retrieval at length** (`needle_gate.py`). A short gate certifies an attention path
   the long-context numbers never execute — and on vLLM the short path is a *different* path
   (see below). The needle sits at 10% depth because divergence lands ~11% in, so a needle at
   the very front can be answered from a prefix a degraded model still holds.

## Every speed prompt is > 2048 tokens, deliberately

vLLM routes MLA prefill with `prefill_max_seq_len <= topk_tokens` (2048) to the dense MHA path
(`mla_attention.py:756`), and on ROCm the selected `ROCM_AITER_MLA_SPARSE` backend implements
`forward_mha` as `raise NotImplementedError` (`v1/attention/backend.py:1040`). A short prompt
therefore **kills the engine**. Forcing the sparse path instead
(`--attention-config '{"sparse_mla_force_mqa":true}'`) runs it outside its design range and
measured **GSM8K 0.190 against plow's 0.970**.

Above 2048 both engines run their intended path and both are correct. That is the only region
where a speed comparison means anything, so `speed.py` keeps every prompt above it.

## Prefix caching is OFF on both

The TTFT ladder sends the same prompt 3×, and GSM8K shares one 8-shot preamble across every
question. With caching on, vLLM pays for those once and plow pays every time — plow has no
prefix cache at all, so leaving it enabled measures a **feature** gap and reports it as a
**kernel** gap. Off is like-for-like. vLLM would gain further with it on; that is a real and
separate advantage and belongs in a writeup as one.

## Usage

```bash
# GSM8K data (this box's IPv6 route to raw.githubusercontent.com is broken -- force IPv4)
mkdir -p ~/.cache/gsm8k && cd ~/.cache/gsm8k
curl -s4LO https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -s4LO https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl

# accuracy + ladder
scripts/twoengine/run_plow.sh <assets> 8231 plow
scripts/twoengine/run_vllm.sh 8241

# speed (throughput + long context), either engine
scripts/twoengine/run_speed.sh plow 8251 plow <assets>
scripts/twoengine/run_speed.sh vllm 8252 vllm
```

Env: `CONCS INLEN OUTLEN NMULT CTXS LC_OUTLEN N SHOTS MAXTOK CONC GSM8K_DIR OUT PLOW_REPO
VLLM_VENV PLOW_NIX PLOW_ROCM_LIB SERVE_ENV MAXLEN GATE_TOK`.

`SERVE_ENV` defaults to `PLOW_MLA_PF_V2=1`: the GLM blob carries the causal KV-split (`ns=2`),
which only the V2 flash arm honours, and the runtime **refuses to load** without it. That
refusal is the arm-refusal chain working — a loud error rather than wrong output — and it is why
`build.json` must travel next to `model.pkt`.
