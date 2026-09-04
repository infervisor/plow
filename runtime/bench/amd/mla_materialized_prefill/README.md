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

### Same-input boundary gate

Whole-model logits do not isolate MLA when an earlier layer already differs.
For promotion, capture the first target attention boundary as generic semantic
tensors. Every reference repeat and both Plow arms must contain byte-identical
`q`, `k`, and `v` payloads (or equivalently named projected inputs), plus the
BF16 attention output and the first downstream residual output. Manifests also
carry the SHA256 of the complete token history and every tensor payload.

Run the local decision gate with adjacent pinned-runtime repeats:

```sh
nix develop -c python3 scripts/mla_boundary_quality_gate.py \
  --reference /tmp/mla/vllm-0.json \
  --reference /tmp/mla/vllm-1.json \
  --reference /tmp/mla/vllm-2.json \
  --absorbed /tmp/mla/plow-absorbed.json \
  --materialized /tmp/mla/plow-materialized.json \
  --input-semantic q,k,v \
  --output-semantic attention.output,residual.output \
  --output /tmp/mla/quality-gate.json
```

The gate rejects a history mismatch, a stale/corrupt payload hash, or any input
payload mismatch before comparing outputs. For rel-L2, max-absolute error, and
cosine loss, materialized Plow must be no worse than absorbed Plow plus the
largest adjacent vLLM repeat floor. This boundary gate is required before the
existing long-continuation and TP8 timing gates; it does not replace them.

### Standalone boundary replay ABI

`scripts/mla_boundary_abi.py` seals a model-independent capture. Its contract is
described entirely by dimensions, causal semantics, layout, and softmax scale.
Every manifest includes exact u32le prompt-token and tensor SHA256 values. The
required source set is `latent.q`, `latent.kv`, `rope.k`, and the Q, KV, and
output projection weights. A materialized capture additionally contains BF16
`projected.q` and `projected.kv`.

The pinned vLLM `MLACommonImpl.forward_mha` is not an `nn.Module`: packed K
and V are local variables immediately passed to the custom attention op, so a
standard module hook cannot call or capture that boundary. The generic hook can
now capture exact BF16 module outputs and module weights. `pack-materialized`
then reconstructs byte-exact token/head-dense Q192/K192/V128 from Q projection,
KV projection, and K-RoPE outside the timed region. This avoids a model-class
patch while retaining a precise ABI for a future custom-op interception.

```sh
nix develop -c python3 scripts/mla_boundary_abi.py seal \
  --spec /tmp/mla/spec.json --output /tmp/mla/sealed.json --require-source
nix develop -c python3 scripts/mla_boundary_abi.py pack-materialized \
  --manifest /tmp/mla/sealed.json --output-dir /tmp/mla/packed
nix develop -c runtime/bench/amd/mla_materialized_prefill/build_replay.sh \
  /tmp/mla/replay
GPU_LEASE_DIR=/tmp/gpulease nix develop -c python3 scripts/mla_boundary_abi.py \
  replay-materialized --manifest /tmp/mla/packed/manifest.json \
  --binary /tmp/mla/replay/replay --object /tmp/mla/replay/opus.elf \
  --gpulease perf-data/tools/gpulease --output-dir /tmp/mla/plow-materialized
GPU_LEASE_DIR=/tmp/gpulease nix develop -c python3 scripts/mla_boundary_abi.py \
  replay-absorbed --manifest /tmp/mla/packed/manifest.json \
  --binary /tmp/mla/replay/replay-absorbed --object /tmp/mla/replay/kernel.co \
  --gpulease perf-data/tools/gpulease --output-dir /tmp/mla/plow-absorbed
```

Both replays perform three adjacent launches and reject any BF16 output-byte drift.
The absorbed replay derives its absorbed Q, Q-RoPE, and value-fold weights once
from the same sealed factor weights and latent tensors, also outside timing.
The materialized replay records the object hash and median kernel time. Input capture, hashing,
packing, uploads, and downloads are outside the event interval. It deliberately
does not synthesize `residual.output`: that tensor must come from each runtime's
real output projection/residual seam. The vLLM capture template reads the second
output of `post_attention_layernorm`, which is the BF16 residual produced after
the attention projection. Plow normally fuses the corresponding AttnRes and
following RMSNorm in place, so a capture asset must be emitted with
`PLOW_K3_FUSE_ARNORM=0 PLOW_SEG_PER_OP=1`; this materializes
`act.l<layer>.h2` and gives it its own segment. Capture that segment with
`PLOW_PF_CAPTURE=T:SEG:act.l<layer>.h2=/tmp/mla/residual.output.bf16`.
The opt-out is diagnostic; production remains fused. Seal both tensors under
`residual.output` with the same prompt history before running the gate. TP8
promotion remains blocked until this real seam and the attention output pass.

Attach an actual captured seam without rewriting any other manifest field:

```sh
nix develop -c python3 scripts/mla_boundary_abi.py attach-tensor \
  --manifest /tmp/mla/plow-materialized/manifest.repeat-0.json \
  --semantic residual.output --file /tmp/mla/residual.output.bf16 \
  --dtype bf16 --shape 8192,7168 --layer 0 --rank 0 \
  --output /tmp/mla/plow-materialized/qualified.repeat-0.json
```

This preserves and revalidates the original prompt, contract, latent, weight,
and Q/K/V hashes. For a full K3 seam, byte-identical MLA inputs are not alone
sufficient: AttnRes also consumes the current prefix, its saved-residual ring,
and learned score weights. Those inputs must either hash identically in the
manifest or be injected from the same capture. Comparing residuals from two
ordinary whole-model runs does not meet this gate when an earlier layer already
differs.
