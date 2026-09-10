# Gemma 3 12B on H100: BF16 and FP8

Measured on 2026-09-10, branch `tp-bringup-mi300x` at `26ebb0ea` plus the
uncommitted Gemma 3 support and prefill slicing fix. This is Gemma **3**, not
the Gemma 4 models in earlier performance records.

This report covers the initial optimization phase. The vLLM comparison through
16K input and concurrency 128 remains in progress; these results do not establish
that competitive target.

## Workload and precision

- H100 80GB HBM3, 132 SMs. One model process on the GPU.
- `unsloth/gemma-3-12b-it`, revision
  `9478e665381f42974aa06177b019352fb6291876`. All five weight shard hashes match
  Google's revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Text decoder: 48 layers, hidden 3840, intermediate 15360, 16 query heads,
  8 KV heads, head dimension 256, vocabulary 262208.
- BF16 baseline and candidate retain BF16 weights and activations.
- FP8 W8A16 uses per-output-row E4M3 projection weights and BF16 activations.
  Norms, embeddings, tied output head and KV remain BF16.
- The separate W8A8 screen also quantizes projection inputs per row to E4M3.
  It changes numerical precision and is not an equal-precision replacement.
- Exact input counts 1024/4096/8192, exactly 128 output tokens, temperature 0,
  ignore EOS, prefix cache disabled, one warmup then five repetitions per case.
  Decode capacity is 1. C4 checks exercise queued serving, not a batch-4 kernel.

## Roofline and selected implementation

The streaming-read probe measured 3162 GB/s. At 1K context, an optimistic
projection/head/KV read floor is 7.57 ms/token for BF16 and 4.17 ms/token for
FP8 weights. Baseline decode is 12.38 and 9.28 ms/token respectively. These
bounds omit scale reads, activations, repeated reads and dispatch overhead.

Prefill reuses each projection weight across hundreds of rows, moving the
projection bottleneck toward compute. BF16 CUDA event attribution at 1K measured
774 ms in GEMM-class segments versus 13 ms in the other/flash class. That
measurement selected GEMM execution as the first optimization target.

Five standalone BF16 GEMM variants were screened on actual 12B shapes. At
M=512, K=3840, the TMA variants reach 432–439 TF/s for N=4096 versus 200 TF/s
for the uniform cp.async variant; for N=15360, they reach 318–320 versus
197 TF/s. The independent FP32 oracle checks passed. These isolated numbers
guided selection; end-to-end serving remains the performance gate.

Enabling TMA alone regressed BF16 prefill. The successful configuration uses
existing dedicated GEMM objects, separate gate/up GEMMs, tensor maps and
prefill segment slicing. CUDA graph submission is included in the final
serving configuration; no independent speedup is attributed to graphs.

The campaign found and fixed a compiler defect: `PLOW_SEG_CLASS_SLICE=1`
also doubled decode work items (33031 to 64843), slowing decode. The option
now applies to flash-prefill programs. A regression test fails before the fix
and passes afterward with the option enabled. Decode returns to 33031 work items.

## BF16 results

Medians; first-token latency includes prefill and the first decode output.

| Input | Baseline TTFT ms | Candidate TTFT ms | TTFT speedup | Baseline/candidate TPOT ms | End-to-end speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 839.41 | 196.89 | 4.26× | 12.377 / 12.374 | 1.36× |
| 4096 | 3240.69 | 855.71 | 3.79× | 12.599 / 12.637 | 1.97× |
| 8192 | 6518.65 | 1985.12 | 3.28× | 12.987 / 12.980 | 2.25× |

Single-layer confirmation at T=128/512/1024: prefill improves from
2.74/8.78/16.55 ms to 0.84/1.64/3.12 ms. Decode remains within 1 µs:
264.93/268.15/274.45 µs versus 264.24/268.22/274.60 µs.

Baseline replay at 1K/8K measured TTFT 833.05/6519.70 ms, within 0.8%
of the original medians. Decode varies by about 1% across these runs;
no decode speedup is attributed to the prefill change.

## FP8 W8A16 results

W8A16 projections still use the ordinary prefill kernels: they have no TMA maps,
so the compiler keeps them on the general segment object. Only the mapped BF16
head uses the dedicated GEMM object. A TMA/WGMMA path for W8A16 projections
remains an optimization target.

The selected FP8 configuration preserves BF16 activations and uses the same
interpreter objects as the BF16 candidate.

| Input | Baseline TTFT ms | Candidate TTFT ms | TTFT speedup | Baseline/candidate TPOT ms | End-to-end speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1323.39 | 350.82 | 3.77× | 9.282 / 9.264 | 1.64× |
| 4096 | 5216.97 | 1484.66 | 3.51× | 9.563 / 9.579 | 2.38× |
| 8192 | 10476.64 | 3237.88 | 3.24× | 9.888 / 9.912 | 2.61× |

A subsequent baseline replay at 1K/8K measured TTFT 1322.20/10473.41 ms,
within 0.1% of the original medians.

Single-layer W8A16 prefill at T=128/512/1024 improves from
6.88/13.89/27.30 ms to 1.77/3.35/6.67 ms. Decode medians are
205.61/206.83/214.92 µs versus 204.58/209.01/215.25 µs.

The separate W8A8 screen achieved TTFT 213.10/896.20/2035.30 ms and TPOT
9.382/9.591/9.928 ms. It is faster, with the numerical tradeoff below; it is
not the selected equal-precision comparison.

At C4 with 1K input and 128 outputs, median aggregate output throughput is
46.25→68.30 tokens/s for BF16 and 44.82→76.53 for W8A16. These are three-repeat
queued-serving screens with visible scheduling variance; use the C1 tables
for the controlled kernel comparison.

## Numerical validation and limits

Native H100 RMSNorm, head norm/RoPE and sandwich norm checks pass at rows
1/4/129, with relative L2 at most 2.6e-5 against independent formulas.
All tested full-model configurations answer the three short smoke prompts
coherently. These checks do not constitute a model quality evaluation.

An independent PyTorch layer-0 calculation on seeded 128×3840 synthetic
input gives relative L2 0.01276 for the BF16 baseline and 0.01231 for the
BF16 candidate. Fused arithmetic differs from eager BF16 rounding; strict
PyTorch parity is not established. The FP8 W8A16 baseline gives 0.04562;
the W8A8 candidate gives 0.08165 against the same BF16 reference. The larger
W8A8 difference is why its speed is reported separately from W8A16.
The selected W8A16 candidate gives 0.04538, preserving the baseline's error
level. Both selected configurations answer the access-code smoke check after
5909 prompt tokens.

Gemma 3 device emission currently requires `sm_90a`; native Gemma 3 norm
support on AMD/CPU is not claimed. Shared CPU/HSA/CUDA all-target compilation
passes, and executor tests pass (325 passed, 25 ignored).
An old interpreter object is rejected before inference with
`norm weight offset mismatch: packet requires 1, object provides 0`.

Broad suites expose two failures in unchanged upstream files:
`packet::slots::tests::table_matches_doc_comments` (IndexSelect slot docs) and
`devgen::emit_config::tests::no_raw_env_reads` (GLM DSA knobs). The remaining
121 packet tests and 407 devgen tests pass. Lean verification was unavailable;
the manifests explicitly record `lean.verified=false`.

## Reproduction

Run terminal commands inside `nix develop`. Build the host tools:

```sh
cargo build --release -p plowc -p plowrt --features plowrt/cuda
```

For BF16, emit with the checkpoint path in `gemma_checkpoint` and output
directory in `gemma_assets`:

```sh
PLOW_SEG_CLASS_SLICE=1 PLOW_SEG_PURE_GEMM=1 \
  target/release/plowc --hf-dir "$gemma_checkpoint" --gpu h100 --arch sm_90a \
  --max-ctx 16384 --tma-gemm --no-glu-fuse \
  --emit-decode-batch-ladder 1 --emit-packed-prefill=false --out "$gemma_assets"

PLOW_BUILD_SEG=1 PLOW_BUILD_GEMM_ONLY=1 PLOW_BUILD_TMA_GEMM=1 \
  PLOW_EXTRA_DEFINES=-DPLOW_NV_GEMMA3=1 \
  bash scripts/build_sm90a_cubin.sh "$gemma_objects/interp_sm90a.cubin"

target/release/plowrt serve --assets "$gemma_assets" --port 8013 \
  --prefix-cache=false --pf-seg-pure 1 --pf-seg-graph \
  --nv-cubin "$gemma_objects/interp_sm90a.cubin" \
  --nv-cubin-pf "$gemma_objects/interp_sm90a_pf.cubin" \
  --pf-seg-dir "$gemma_objects"
```

FP8 weights are exported with `perf-data/tools/quantize_fp8.py`, using prefix
`language_model.model.`. The runtime checkpoint directory must contain both
the original shards and the FP8 twin. Add `--fp8 --w8a16` when emitting and
`--rt-checkpoint "$gemma_fp8_checkpoint"` when serving. The W8A8 experiment
instead uses `--fp8 --w8a8` and builds with `PLOW_BUILD_W8A8=1`.

```sh
python3 scripts/bench_packed_serve.py --url http://127.0.0.1:8013 \
  --out "$gemma_results" --label gemma3-h100 \
  --inputs 1024 4096 8192 --outputs 128 --concurrency 1 --repeats 5 --warmups 1
```

The independent layer reference is `gemma3-12b-h100/block_reference.py`.
Pass the working artifact directory as its argument; it reads `checkpoint/`
and `block-input.npy`. The input was NumPy `default_rng(42).normal(0, 1,
(128, 3840)).astype(float32)`. It requires PyTorch, NumPy and safetensors.

Selected raw results, kernel measurements and artifact hashes are in
[`gemma3-12b-h100/`](gemma3-12b-h100/). The `.json` request-result files contain
one JSON object per line. Working artifacts, checkpoint metadata, independent reference code, logs and
packet/object files are in `plans/gemma3-12b-roofline/` in this workspace.
