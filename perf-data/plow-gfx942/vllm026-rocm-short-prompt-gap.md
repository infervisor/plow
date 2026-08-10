# vLLM 0.26 on ROCm cannot serve GLM-5.2 below 2048 tokens

> **Scope:** vLLM 0.26.0 / torch 2.11.0 / ROCm 7.14 on 8x MI300X (gfx942, CDNA3, 304 CU, 8 XCDs, 64 KiB LDS, ROCm 7.2.4) · **EXTERNAL** — a property of vLLM's ROCm sparse-MLA path, not of plow or of the GPU. Would need re-checking against any other vLLM build.

**Status: measured 2026-08-09, one session, this box.** Stack: vLLM 0.26.0, torch
2.11.0+gitd0c8b1f, ROCm 7.14 / gfx942, 8× MI300X, `/workspace/vllm26`, `VLLM_ROCM_USE_AITER=1`.

This is a **capability** finding, not a benchmark result, and it constrains every speed number
this campaign can publish. Read §4 before quoting §3.

---

## 1. The failure

GLM-5.2 is a DSA model, so vLLM selects the sparse MLA backend — and the log shows it is the
**only** candidate:

```
rocm.py:624    Using ROCM_AITER_MLA_SPARSE backend out of potential backends: ['ROCM_AITER_MLA_SPARSE']
selector.py:174 Using ROCM_AITER_FA MLA prefill backend.
```

`mla_attention.py:756` then decides, per batch, whether prefill takes the dense MHA path:

```python
if self.impl.is_sparse and num_mha_tokens > 0:
    use_mha = (self.prefill_backend is not None
               and prefill_max_seq_len <= attn_metadata.topk_tokens      # 2048 for GLM-5.2
               and not self._vllm_config.attention_config.sparse_mla_force_mqa)
```

So **short** sequences are routed to dense MHA deliberately — the sparse path is meant for
sequences longer than `top_k`. But that backend's MHA entry point is unimplemented:

```
v1/attention/backend.py:1040    def forward_mha(...): raise NotImplementedError
```

**Any prompt under 2048 tokens kills the engine.** The coherence gate — literally *"What is the
capital of France?"* — took the whole server down on the first request:

```
core.py:1332  EngineCore encountered a fatal error.
async_llm.py:704  vllm.v1.engine.exceptions.EngineDeadError
POST /v1/chat/completions HTTP/1.1  500 Internal Server Error
```

---

## 2. The workaround, and what it costs

vLLM's own escape hatch forces every token down the sparse path:

```
--attention-config '{"sparse_mla_force_mqa":true}'
```

The engine then serves. But this runs the sparse path **outside its design range** — vLLM picks
MHA for short sequences on purpose. Measured cost, GSM8K 8-shot CoT, greedy, exact match on the
last number (lm-eval convention), n=100, **identical client for both engines**
(`scripts/twoengine/client.py`), prefix caching off on both:

| engine | exact match | errors | wall | throughput |
|---|--:|--:|--:|--:|
| **plow** (gfx942, TP8) | **0.9700** | 0 | 317 s | 0.316 q/s |
| **vLLM 0.26 + AITER**, `force_mqa` | **0.1900** | 0 | 577 s | 0.173 q/s |

GSM8K prompts here are ~1k tokens, i.e. entirely inside the broken region.

---

## 3. What this is, and what it is NOT

**It is not evidence that vLLM is inaccurate.** vLLM is an excellent engine and the 0.190 is an
artefact of being forced off its intended code path.

**It is evidence that on this ROCm stack, at ≤2048-token prompts, vLLM 0.26 has no correct path
for GLM-5.2** — it either dies or degrades — while plow serves the same regime at 0.970. That is
a real capability difference and it is the first accuracy comparison this project has ever had
for GLM. Every prior gate was token-identity (`k3_tp_equivalence.sh`, the Paris continuation) or
the facts gate; those prove self-consistency, not correctness.

Scope it honestly when quoting: it is a claim about **this build**, on **this stack**, for **this
model family**. Re-check on a newer vLLM before saying it anywhere external — it may be fixed
upstream.

---

## 4. The constraint this places on every speed comparison

**No short-prompt vLLM number from this box is publishable.** Either the engine is dead or it is
computing something different from what plow computes, and a ratio between two different
computations is not a speedup.

So all speed benching moves **above 2048 tokens**, where both engines run their intended path and
both are correct. `scripts/twoengine/speed.py` enforces this and says so in its header, and the
standing gate is needle retrieval at length (`needle_gate.py`) rather than a short liveness
check — a short gate certifies a path the long-context numbers never execute.

plow passes that gate: needle at 10% depth of a 3000-token prompt, answered `7413` exactly.

---

## 5. Reproducer

```bash
# dies on the first short request
/workspace/vllm26/bin/vllm serve /workspace/models/GLM-5.2-FP8 \
  --served-model-name glm-5.2-fp8 --tensor-parallel-size 8 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --trust-remote-code --port 8241
curl -s localhost:8241/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-fp8","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
# -> 500, EngineDeadError

# serves, but degraded on short prompts
#   add: --attention-config '{"sparse_mla_force_mqa":true}'
```

Raw results: `perf-data/plow-gfx942/r2-baseline/{plow,vllm-force-mqa}.json`.

---

## See also

- `scripts/twoengine/README.md` — the one-client harness and why it exists
- `LESSONS.md` §3 — a gate that has never gone red is not evidence of anything. This one went red
  on its first real use, on both engines, for two different reasons (plow: CPU-backend fallback;
  vLLM: engine death).
