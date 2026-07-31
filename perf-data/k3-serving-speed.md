# K3 serving: speed and TTFT through `plowrt serve`, on the shipped decode stack

**Measured 2026-07-30**, Kimi-K3 93 layers TP8 over 8× gfx950 (MI355X), fp8 KV + mxfp4 experts,
blob and objects emitted from this tree with **`PLOW_GATE_HIER` + `PLOW_L2_PLACE` +
`PLOW_K3_SHARD_HEAD`**, under `perf-data/harness/gpulease`. Client is `scripts/bench_speed.sh`.

**§0-BENCH.** `bench_plowrt_serve.sh` is the REFERENCE harness and is preferred whenever it can
run, because it drives `vllm bench serve` so plow and vLLM are measured by the same client binary
with the same metric definitions. It needs the `rocm/vllm:...` image, which this box does not
have. `bench_speed.sh` is the fallback: same metric definitions, no docker, streaming so TTFT is
real. **Nothing here may be tabled next to a vLLM number** — different client, unvalidated against
the reference. It measures plow against itself, which is what a regression gate needs.

## 1. Decode, end to end

| in_tok | conc | n | TTFT mean | TTFT med | **TPOT** | ITL p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 6 | 352.1 ms | 345.9 | **27.52 ms** | 27.79 |
| 1024 | 1 | 6 | 939.8 | 887.0 | **31.57** | 219.83 |
| 4096 | 1 | 6 | 3699.3 | 2639.3 | **32.27** | 312.88 |
| 8192 | 1 | 3 | 4705.5 | 4572.0 | 28.24 | 28.43 |
| 16384 | 1 | 3 | 9683.5 | 9692.1 | 29.79 | 52.98 |
| 30000 | 1 | 3 | 19175.1 | 19262.4 | 29.82 | 30.07 |

**TPOT 27.5–32.3 ms = 31–36 tok/s per stream**, and it CROSS-VALIDATES the kernel-level number:
`amd-bench` measures 28.876 ms at ctx 32000 on the same stack, and the serving path measures
29.82 ms at 30000 in-tokens. The two instruments agree to 3%, which is worth having because they
share no code path above the engine.

`conc` is 1 throughout on purpose. K3 decode is **structurally single-sequence** — the KDA
recurrent state has no batch axis (`exec/amd.rs:3264`) and `AmdTpGroup::submit_decode` is scalar
(`serve/engine.rs:187`) — so a concurrency column here would measure QUEUEING, not batching, and
the harness says so itself rather than letting a reader assume otherwise.

## 2. TTFT is LINEAR in prompt length, and that is the chunker working

```
   128 ->  0.35 s        8192 ->  4.57 s
  1024 ->  0.89 s       16384 ->  9.69 s
  4096 ->  2.64 s       30000 -> 19.26 s
```

~0.63 s per 1000 prompt tokens, flat across a 234× range. `plan_chunks`' bucket-ladder DP is
doing its job; nothing here is quadratic.

## 3. THE PREFILL RESULT IS NOT WHAT THE KERNEL NUMBER IMPLIES, and this is the finding

`k3-mla-prefill-mfma` measures **2.25–2.79×** on the MLA prefill kernel and 7.2–8.4× on bf16.
Those numbers are real and reproduced. But end to end:

```
  kimi-k3-README.md, pre-MFMA:  ~24 s TTFT at 32k
  measured here, post-MFMA:      19.26 s at 30000
```

**~20%, not 2.25×.** The reason is arithmetic, not a regression: **the MFMA kernel covers the 24
MLA layers and K3 has 69 KDA layers**, and `kimi-k3-kernel-gap.md:847` still lists KDA chunked
prefill as NOT DONE — those layers run prefill-as-N-decodes. Amdahl caps the whole model at the
fraction MLA occupies.

So the prefill lever is **not** more MLA kernel work. It is KDA chunked prefill, and until that
lands, TTFT is dominated by 69 layers that the merged win does not touch. Anyone quoting
"2.25–2.79× faster prefill" as a model-level claim is quoting a kernel number as an end-to-end one.

## 4. Reproduce

```bash
perf-data/harness/gpulease -n 8 speed sg render -c \
  "PLOW_L2_PLACE_DISPATCH=1 IN_LENS='128 1024 4096' CONCS=1 NPROMPT=6 OUTLEN=128 \
   PLOWRT_BIN=<hsa-built plowrt> scripts/bench_speed.sh <assets> 8421 auto 1800"
```

`<assets>` needs `model.pkt`, `hsaco/`, `checkpoint/`, `tokenizer.json` **and `weights.json`** —
the last one is easy to miss and the server exits with a bare `Io { NotFound }` rather than a
diagnosis.
