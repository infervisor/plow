# Materialized MLA prefill screen

This harness compares Plow's absorbed attention plus fold, its first rectangular
materialized kernel, and the generic gfx950 `D_QK=192`, `D_V=128` Opus schedule.

The measured schedule is vendored as a standalone runtime object under
`runtime/amd/third_party/aiter_opus`. It is based on AITER upstream commit
`10b192f5b5bda90f2af33ceae7a6c2f416bfc674` and retains the MIT license. The
runtime object has no build-time or run-time AITER dependency. A guarded Plow
adaptation decodes the original 3D batch grid from plowrt's flat 1D launch.

The useful schedule is selected by dimensions and architecture, not a model name:

- wave64, eight waves, 32 query rows per wave;
- 64 KV rows per stage;
- full Q fragments retained in registers;
- two K and two V LDS buffers, with V reusing Q's dead LDS lifetime;
- score and value GEMMs split into two super-units;
- a 16-stage K/V/MFMA/softmax pipeline with staggered wave groups.

Run with:

```sh
nix develop -c env SAMPLES=31 \
  runtime/bench/amd/mla_materialized_prefill/run.sh /tmp/plow-mla-opus-gate
```

The script rejects spilling standalone objects through explicit compiler-resource
checks. It compares the flat object byte-for-byte with a 3D-grid oracle for every
head, including a ragged 1025-token launch. Exact 256-token multiples must also
remain within max absolute error `0.02` and RMSE `0.003` against the absorbed form.
The full-path timing includes both sides' query projection GEMMs; the materialized
side additionally includes KV projection and packing.

The full-path oracle derives the absorbed BF16 query and value weights from the same
factor weights used by the materialized side. This is distinct from the kernel-only
oracle above: independently initialized factor and absorbed weights can time the
projection pipelines, but cannot establish their numerical equivalence.

The old in-tree `k_materialized` comparator is retained as a diagnostic but is not a
promotion gate: its output is nondeterministically non-finite on gfx950. Set
`PLOW_MLA_LEGACY_GATE=1` to reproduce its historical hard gate, or
`PLOW_MLA_DIAGNOSTICS=1` to print finite counts after every stage. The production Opus
object and the consistent-weight full-path oracle remain hard gates.

## Real-weight exactness investigation

The consistent-weight synthetic oracle is useful for wiring, but its factor
magnitudes made the final delta artificially small. With one real Kimi-K3
checkpoint layer, the factorized and absorbed paths first differ at layer 3's
attention output. An env-gated, rank-0, one-shot capture at T8192 measured:

| comparison | BF16 mismatches | max abs | RMSE |
|---|---:|---:|---:|
| materialized Q, projection vs CPU reference | 1,679 / 18,874,368 | 0.0625 | 6.67823e-5 |
| materialized KV, projection vs CPU reference | 2,910 / 25,165,824 | 0.00390625 | 3.21216e-6 |
| packed K, captured projection vs attention input | 0 / 18,874,368 | 0 | 0 |
| packed V, captured projection vs attention input | 0 / 12,582,912 | 0 | 0 |
| materialized vs absorbed attention output | 2,426,661 / 12,582,912 | 0.00390625 | 8.69558e-5 |
| post-attention residual seam | 14,693,882 / 58,720,256 | 0.046875 | 1.38852e-4 |

The materialized Q capture also persisted byte-for-byte from projection segment
8 to attention segment 10. Re-deriving `q_absorb`, `q_rope`, and `v_absorb` from
the raw checkpoint factors was bit-exact across all 24 MLA layers and all 96
heads. Weight provenance, TP slicing, transient routing, and K/V packing are
therefore ruled out. The alternative BF16 contraction order is the first
observed difference, and the residual path amplifies it.

The corrected same-weight full-path synthetic oracle remains finite and small:
max abs `3.8147e-6`, RMSE `2.00983e-7` at T1024; max abs `3.8147e-6`, RMSE
`1.19313e-7` at T8192; and max abs `3.8147e-6`, RMSE `2.00885e-7` at ragged
T1025. Its total projection-plus-attention timing was 2.820 vs 1.653 ms at T1024
and 12.958 vs 4.077 ms at T8192. These values validate the harness, not
real-checkpoint equivalence.

FP32 transient storage alone cannot repair the result because the BF16 attention
kernel rounds at its inputs. A true FP32 attention schedule would double the
T8192 Q/K/V transient from 144 to 288 MiB and its analogous tile needs about
300 KiB LDS, above gfx950's 160 KiB workgroup limit. Reproducing the absorbed
contraction order reconstructs about 3.45x the materialized attention work and
removes the measured gain. Neither is an exactness repair worth promoting. The
materialized route remains default-off.

## Pinned vLLM baseline path

The retained vLLM 0.28 baseline is source commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. Its K3 AMD layer passes BF16
`q_b_proj` output directly as Q. `MLACommonBaseImpl::forward_mha` runs BF16
`kv_b_proj`, splits the result into K-nope and V, concatenates the shared K
position component, and calls `ROCM_AITER_FA`. The baseline log confirms the
loaded gfx950 object is
`fmha_fwd_hd192_hd128_bf16_causal_group`.

This is the compute-friendly materialized algorithm, not absorbed latent math:
Q and K are 192-wide and V is 128-wide. Projection MFMAs accumulate into FP32
and round their outputs to BF16; the AITER kernel consumes those BF16 Q/K/V,
uses BF16 MFMA products with FP32 score/value accumulators and softmax state,
then rounds its output to BF16. The Python path specifies these precision
boundaries, but the selected GEMM algorithms and hand-written AITER object own
the reduction trees. Their instruction-level order is not a portable contract.
Consequently, vLLM itself follows the same numerically alternative formulation
as this candidate and is not expected to match Plow's absorbed path bitwise.

## External correctness gate

A future promotion gate should compare the candidate to the retained vLLM
baseline rather than require equality with a different contraction graph:

1. Pin checkpoint hashes, tokenizer files, vLLM commit/container, AITER object,
   Plow commit/assets, BF16 KV, TP8, and all sampling parameters.
2. Use a fixed corpus containing short prompts, exact 1024/8192 prompts, a
   ragged prompt, and continuation prompts. Require both `/tokenize` endpoints
   to return identical prompt token IDs before inference.
3. Run non-streaming greedy `/v1/completions` with EOS ignored and fixed output
   lengths of 1 and 256. Plow's `return_token_ids` records exact prompt and
   completion IDs; obtain vLLM IDs from its completion response/tokenizer.
   Require identical generated IDs at every position across three cold starts.
4. Add a diagnostic-only, rank-0, one-shot logits download to Plow before the
   logits part of the gate. Compare the same teacher-forced token prefix at
   every step, requiring finite logits, identical top-1, and bounded max-abs,
   RMSE, and top-1 margin error. Record full-logit hashes and the first failing
   step. Do not infer logits parity from decoded text or top-k API logprobs.

Plow currently returns token IDs but not real GPU logits from the OpenAI API, so
step 4 is a prerequisite for a genuine external logits gate. Until that exists
and passes, the 130 ms TTFT result remains experimental and the route stays off.

The CPU-side token gate is `scripts/openai_correctness_gate.py`. It never starts
a server, and its corpus preparation and capture commands do no network I/O
unless `--execute` is present. Prepare one fixed exact-length corpus against a
canonical tokenizer, then restart each server before each capture:

```sh
nix develop -c python3 scripts/openai_correctness_gate.py prepare-corpus \
  --base-url http://127.0.0.1:8000 --model MODEL --output /tmp/corpus.json
nix develop -c python3 scripts/openai_correctness_gate.py capture \
  --base-url http://127.0.0.1:8000 --model MODEL --arm plow --run-id 1 \
  --corpus /tmp/corpus.json --output /tmp/plow-1.json
```

The commands above only print their execution plans. Add `--execute` explicitly
after the appropriate server is already running. Capture run IDs 1, 2, and 3
for each arm, with a cold server start before every invocation, then compare:

```sh
nix develop -c python3 scripts/openai_correctness_gate.py compare \
  --left /tmp/plow-{1,2,3}.json --right /tmp/vllm-{1,2,3}.json
```

The comparator requires exact cold-run determinism within each arm, then exact
prompt and output token parity between arms. It reports the first prompt token
or generated-token divergence.
